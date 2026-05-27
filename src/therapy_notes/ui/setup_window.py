from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QObject
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QButtonGroup,
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from ..audio.recorder import AudioRecorder
from ..config import config
from ..transcription.transcriber import Transcriber
from ..utils.env import save_env_value

# ── Test constants ────────────────────────────────────────────────────────────

# Script shown to the therapist to read aloud during the mic test.
# Chosen for phonetic variety and natural cadence.
_MIC_SCRIPT = (
    "Good morning. Testing the audio recording system. "
    "My name is Sarah and I am conducting a telehealth session today. "
    "One, two, three, four, five."
)

# Phrase spoken by macOS `say` during the loopback test.
_LOOPBACK_PHRASE = "Testing system audio capture. One two three four five. Audio check complete."

_MATCH_THRESHOLD = 0.50   # 50% word overlap = pass
_MIC_RECORD_SECONDS = 10


# ── Helpers ───────────────────────────────────────────────────────────────────

def _word_match(expected: str, actual: str) -> float:
    """Return fraction of expected words found anywhere in actual."""
    clean = lambda s: re.sub(r"[^\w\s]", "", s.lower())
    exp = set(clean(expected).split())
    act = set(clean(actual).split())
    if not exp:
        return 0.0
    return len(exp & act) / len(exp)


def _blackhole_loaded() -> bool:
    """True if BlackHole 2ch is visible as an active audio device."""
    try:
        from ..audio.devices import find_device
        return find_device("BlackHole 2ch", kind="input") is not None
    except Exception:
        return False


def _blackhole_installed() -> bool:
    """True if the BlackHole HAL driver file exists on disk (may not be loaded yet)."""
    return Path("/Library/Audio/Plug-Ins/HAL/BlackHole2ch.driver").exists()


def _homebrew_available() -> bool:
    return shutil.which("brew") is not None


def _api_key_ok() -> bool:
    if config.ai_provider == "anthropic":
        return bool(config.anthropic_api_key)
    return bool(config.openai_api_key)


def _spacy_model_installed() -> bool:
    try:
        import spacy
        return spacy.util.is_package("en_core_web_lg")
    except Exception:
        return False


# ── Per-test signals (must be class-level attrs on a QObject subclass) ────────

class _TestSignals(QObject):
    result = pyqtSignal(str, float)   # transcript, match_ratio
    error  = pyqtSignal(str)


class _OllamaSignals(QObject):
    ok    = pyqtSignal(str)   # success — model reply preview
    error = pyqtSignal(str)   # failure message


class _SpacySignals(QObject):
    ok    = pyqtSignal()
    error = pyqtSignal(str)


class _UpdateSignals(QObject):
    ok    = pyqtSignal(str)   # summary of what changed
    error = pyqtSignal(str)



# ── Setup window ──────────────────────────────────────────────────────────────

class SetupWindow(QDialog):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Therapy Notes — Setup & Test")
        self.setMinimumSize(660, 560)
        self.resize(660, 720)

        self._mic_countdown = _MIC_RECORD_SECONDS
        self._mic_timer = QTimer()
        self._mic_timer.setInterval(1000)
        self._mic_timer.timeout.connect(self._mic_tick)
        self._mic_recorder: AudioRecorder | None = None

        self._build_ui()
        # Auto-check silent items on open
        QTimer.singleShot(100, self._auto_check)

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        outer.addWidget(scroll)

        container = QWidget()
        scroll.setWidget(container)
        self._main = QVBoxLayout(container)
        self._main.setContentsMargins(24, 20, 24, 24)
        self._main.setSpacing(10)

        hdr = QLabel("Setup & Test")
        hdr.setFont(QFont("Helvetica", 18, QFont.Weight.Bold))
        self._main.addWidget(hdr)

        sub = QLabel("Complete these steps once before using the app for the first time.")
        sub.setStyleSheet("color: gray; font-size: 13px;")
        self._main.addWidget(sub)
        self._main.addSpacing(4)

        # ── Step 1: BlackHole ─────────────────────────────────────────────────
        self._bh = self._step(
            "Step 1 — BlackHole Audio Driver",
            "BlackHole 2ch is a free virtual audio driver that lets this app hear "
            "what comes through your speakers — i.e. the client's voice from Zoom "
            "or Google Meet.",
            btn="Check / Install",
        )
        self._bh.btn.clicked.connect(self._handle_blackhole)

        # ── Step 2: Multi-Output Device ───────────────────────────────────────
        self._mo = self._step(
            "Step 2 — Multi-Output Device  (manual, ~2 min)",
            "Complete Step 1 first — BlackHole 2ch must appear in the left "
            "panel of Audio MIDI Setup before you can continue here.\n\n"
            "  1. Click the button below to open Audio MIDI Setup.\n"
            "  2. Look for 'BlackHole 2ch' in the left panel. If it's missing, "
            "finish Step 1 and restart your Mac.\n"
            "  3. Click + at the bottom-left → 'Create Multi-Output Device'.\n"
            "  4. Check 'MacBook Pro Speakers' (or your headphones) AND "
            "'BlackHole 2ch'.\n"
            "     ⚠ Do NOT check 'ZoomAudioDevice' — that is Zoom's internal "
            "device and is not the same as BlackHole.\n"
            "  5. Rename it something clear, e.g. 'Therapy Output'.\n"
            "  6. In Zoom → Settings → Audio → Speaker, choose 'Therapy Output'.\n"
            "     Google Meet uses the system default automatically — no change needed.",
            btn="Open Audio MIDI Setup",
            status_char="⚙",
        )
        self._mo.btn.clicked.connect(self._open_audio_midi)

        # ── Step 3: AI Provider ───────────────────────────────────────────────
        self._api = self._step(
            "Step 3 — AI Provider",
            "Choose which AI generates session notes.\n"
            "Ollama runs entirely on this Mac — no data is ever sent anywhere.",
            btn="Check",
        )
        self._api.btn.clicked.connect(self._check_provider)

        # Provider radio buttons
        self._provider_group = QButtonGroup(self)
        self._rb_anthropic = QRadioButton("Anthropic (Claude)")
        self._rb_openai = QRadioButton("OpenAI (GPT-4o)")
        self._rb_ollama = QRadioButton("Ollama — local, no data sent")
        rb_row = QHBoxLayout()
        for rb in (self._rb_anthropic, self._rb_openai, self._rb_ollama):
            rb_row.addWidget(rb)
            self._provider_group.addButton(rb)
        rb_row.addStretch()
        rb_widget = QWidget()
        rb_widget.setLayout(rb_row)
        self._api.col.addWidget(rb_widget)

        # API key row (Anthropic / OpenAI)
        key_row = QHBoxLayout()
        self._api_key_input = QLineEdit()
        self._api_key_input.setEchoMode(QLineEdit.EchoMode.Password)
        self._api_key_input.setPlaceholderText("sk-ant-…  or  sk-…")
        self._api_key_input.setMinimumWidth(220)
        key_row.addWidget(self._api_key_input)
        self._api_show_btn = QPushButton("Show")
        self._api_show_btn.setFixedWidth(54)
        self._api_show_btn.setCheckable(True)
        self._api_show_btn.toggled.connect(
            lambda on: self._api_key_input.setEchoMode(
                QLineEdit.EchoMode.Normal if on else QLineEdit.EchoMode.Password
            )
        )
        key_row.addWidget(self._api_show_btn)
        self._api_save_btn = QPushButton("Save to .env")
        self._api_save_btn.clicked.connect(self._save_api_key)
        key_row.addWidget(self._api_save_btn)
        self._api_key_widget = QWidget()
        self._api_key_widget.setLayout(key_row)
        self._api.col.addWidget(self._api_key_widget)

        # Ollama row (model name)
        ollama_row = QHBoxLayout()
        ollama_row.addWidget(QLabel("Model:"))
        self._ollama_model_input = QLineEdit()
        self._ollama_model_input.setPlaceholderText("llama3.1")
        self._ollama_model_input.setText(config.ollama_model)
        ollama_row.addWidget(self._ollama_model_input)
        ollama_row.addStretch()
        self._ollama_widget = QWidget()
        self._ollama_widget.setLayout(ollama_row)
        self._api.col.addWidget(self._ollama_widget)

        # Wire provider radio buttons — toggled fires for both check and uncheck,
        # so guard with `on` to only react once per change.
        self._rb_anthropic.toggled.connect(
            lambda on: on and self._on_provider_change("anthropic")
        )
        self._rb_openai.toggled.connect(
            lambda on: on and self._on_provider_change("openai")
        )
        self._rb_ollama.toggled.connect(
            lambda on: on and self._on_provider_change("ollama")
        )

        # Indeterminate progress bar — visible only while Ollama model test runs
        self._api_progress = QProgressBar()
        self._api_progress.setRange(0, 0)
        self._api_progress.setFixedHeight(4)
        self._api_progress.setTextVisible(False)
        self._api_progress.setVisible(False)
        self._api.col.addWidget(self._api_progress)

        # Set initial radio state from config (no signal emitted yet)
        p = config.ai_provider
        if p == "openai":
            self._rb_openai.setChecked(True)
        elif p == "ollama":
            self._rb_ollama.setChecked(True)
        else:
            self._rb_anthropic.setChecked(True)
        self._update_provider_ui()  # show/hide correct input row

        # ── Step 4: spaCy Language Model ──────────────────────────────────────
        self._spacy = self._step(
            "Step 4 — Anonymization Model",
            "Downloads the English NLP model (en_core_web_lg, ~560 MB) that "
            "Presidio uses to detect and redact names, locations, and other "
            "identifying information from session transcripts. Only needed once.",
            btn="Check / Download",
        )
        self._spacy.btn.clicked.connect(self._check_spacy)

        self._spacy_progress = QProgressBar()
        self._spacy_progress.setRange(0, 0)
        self._spacy_progress.setFixedHeight(4)
        self._spacy_progress.setTextVisible(False)
        self._spacy_progress.setVisible(False)
        self._spacy.col.addWidget(self._spacy_progress)

        # ── Step 5: Microphone Test ───────────────────────────────────────────
        script_preview = f'"{_MIC_SCRIPT}"'
        self._mic = self._step(
            f"Step 5 — Microphone Test  ({_MIC_RECORD_SECONDS}s recording)",
            f"Read this aloud when the recording starts:\n\n{script_preview}\n\n"
            "Whisper will transcribe the recording and check how well it matches.\n\n"
            "⚠ First run only: the Whisper model (~250 MB) will download before "
            "transcription begins. This is a one-time step — subsequent tests and "
            "sessions will be fast.",
            btn=f"Record & Test ({_MIC_RECORD_SECONDS}s)",
        )
        self._mic.btn.clicked.connect(self._start_mic_test)

        # ── Step 6: Loopback / System Audio Test ──────────────────────────────
        self._lb = self._step(
            "Step 6 — System Audio Capture Test",
            "Plays a 440 Hz tone directly to your 'Therapy Output' Multi-Output "
            "Device and checks that BlackHole 2ch captures it.\n\n"
            "BlackHole will be saved to your .env automatically when you run this "
            "test. The 'Therapy Output' device from Step 2 must exist for this to pass.",
            btn="Run Test",
        )
        self._lb.btn.clicked.connect(self._run_loopback_test)

        # ── Zoom note ─────────────────────────────────────────────────────────
        zoom = QFrame()
        zoom.setFrameShape(QFrame.Shape.StyledPanel)
        zoom_l = QVBoxLayout(zoom)
        zoom_l.setContentsMargins(14, 10, 14, 10)
        zoom_l.addWidget(QLabel("ℹ  About Zoom"))
        note = QLabel(
            "Zoom cannot be tested automatically, but Step 5 validates the same "
            "audio path. If Step 5 passes, you know:\n"
            "  • BlackHole is installed and receiving audio\n"
            "  • Your speakers are routing through BlackHole\n"
            "  • Whisper transcription is working\n\n"
            "The one manual step for Zoom: open Zoom → Settings → Audio → "
            "Speaker and set it to your Multi-Output Device. Google Meet follows "
            "the system default automatically, so Step 2 is sufficient."
        )
        note.setWordWrap(True)
        note.setStyleSheet("color: gray; font-size: 12px;")
        zoom_l.addWidget(note)
        self._main.addWidget(zoom)

        # ── Update App ────────────────────────────────────────────────────────
        self._update = self._step(
            "Update App",
            "Pull the latest changes from GitHub and reinstall dependencies.",
            btn="Check for Updates",
            status_char="↑",
        )
        self._update.btn.clicked.connect(self._run_update)

        self._update_progress = QProgressBar()
        self._update_progress.setRange(0, 0)
        self._update_progress.setFixedHeight(4)
        self._update_progress.setTextVisible(False)
        self._update_progress.setVisible(False)
        self._update.col.addWidget(self._update_progress)

        self._main.addStretch()

    # ── Step card factory ─────────────────────────────────────────────────────

    class _Step:
        """Thin handle returned by _step() for each wizard row."""
        __slots__ = ("status", "btn", "result", "col")

        def __init__(
            self,
            status: QLabel,
            btn: QPushButton,
            result: QLabel,
            col: QVBoxLayout,
        ) -> None:
            self.status = status
            self.btn = btn
            self.result = result
            self.col = col   # the column layout — extra widgets can be added here

        def set_ok(self, text: str = "") -> None:
            self.status.setText("✓")
            self.status.setStyleSheet("color: #27ae60; font-size: 16px;")
            if text:
                self.result.setText(text)
                self.result.setStyleSheet("color: #27ae60; font-size: 12px;")

        def set_fail(self, text: str = "") -> None:
            self.status.setText("✗")
            self.status.setStyleSheet("color: #e74c3c; font-size: 16px;")
            if text:
                self.result.setText(text)
                self.result.setStyleSheet("color: #e74c3c; font-size: 12px;")

        def set_warn(self, text: str = "") -> None:
            self.status.setText("⚠")
            self.status.setStyleSheet("color: orange; font-size: 16px;")
            if text:
                self.result.setText(text)
                self.result.setStyleSheet("color: orange; font-size: 12px;")

        def set_info(self, text: str) -> None:
            self.result.setText(text)
            self.result.setStyleSheet("color: gray; font-size: 12px;")

    def _step(
        self,
        title: str,
        description: str,
        btn: str = "",
        status_char: str = "○",
    ) -> _Step:
        frame = QFrame()
        frame.setFrameShape(QFrame.Shape.StyledPanel)
        row = QHBoxLayout(frame)
        row.setContentsMargins(14, 10, 14, 10)
        row.setSpacing(10)

        status_lbl = QLabel(status_char)
        status_lbl.setFixedWidth(22)
        status_lbl.setFont(QFont("Helvetica", 16))
        status_lbl.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignHCenter)
        row.addWidget(status_lbl)

        col = QVBoxLayout()
        col.setSpacing(4)

        title_lbl = QLabel(title)
        title_lbl.setFont(QFont("Helvetica", 13, QFont.Weight.Bold))
        col.addWidget(title_lbl)

        desc_lbl = QLabel(description)
        desc_lbl.setWordWrap(True)
        desc_lbl.setStyleSheet("color: gray; font-size: 12px;")
        col.addWidget(desc_lbl)

        result_lbl = QLabel("")
        result_lbl.setWordWrap(True)
        result_lbl.setStyleSheet("font-size: 12px;")
        result_lbl.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse |
            Qt.TextInteractionFlag.TextSelectableByKeyboard
        )
        col.addWidget(result_lbl)

        row.addLayout(col, 1)

        btn_widget = QPushButton(btn)
        btn_widget.setFixedWidth(190)
        btn_widget.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        if not btn:
            btn_widget.setVisible(False)
        row.addWidget(btn_widget, 0, Qt.AlignmentFlag.AlignTop)

        self._main.addWidget(frame)
        return self._Step(status_lbl, btn_widget, result_lbl, col)

    # ── Auto-check (runs on open) ─────────────────────────────────────────────

    def _auto_check(self) -> None:
        self._refresh_blackhole()
        self._check_provider()
        self._check_spacy()

    # ── Step 1: BlackHole ─────────────────────────────────────────────────────

    def _refresh_blackhole(self) -> None:
        if _blackhole_loaded():
            self._bh.set_ok("BlackHole 2ch is installed and loaded.")
        elif _blackhole_installed():
            # Driver file is on disk but Core Audio hasn't loaded it
            self._bh.set_warn(
                "BlackHole is installed but the audio server hasn't loaded it yet.\n\n"
                "Click 'Restart Core Audio' below — this reloads the audio server "
                "in about 2 seconds without requiring a full restart.\n"
                "If that doesn't work, restart your Mac and click Check again."
            )
            self._bh.btn.setText("Restart Core Audio")
            self._bh.btn.setEnabled(True)
            self._bh.btn.clicked.disconnect()
            self._bh.btn.clicked.connect(self._restart_coreaudio)
        else:
            self._bh.set_fail(
                "BlackHole 2ch is not installed. Click Check / Install to proceed."
            )
            self._bh.btn.setText("Check / Install")
            self._bh.btn.setEnabled(True)
            self._bh.btn.clicked.disconnect()
            self._bh.btn.clicked.connect(self._handle_blackhole)

    def _handle_blackhole(self) -> None:
        if _blackhole_installed():
            self._bh.set_ok("Already installed.")
            return

        if _homebrew_available():
            # brew install requires sudo for the .pkg step, which needs an
            # interactive terminal to accept a password. Open Terminal so the
            # user can see the prompt and type their password normally.
            subprocess.run(
                [
                    "osascript",
                    "-e", 'tell application "Terminal" to activate',
                    "-e", 'tell application "Terminal" to do script "brew install blackhole-2ch"',
                ],
                check=False,
            )
            self._bh.status.setText("⏳")
            self._bh.status.setStyleSheet("color: orange;")
            self._bh.set_info(
                "A Terminal window has opened running:\n"
                "    brew install blackhole-2ch\n\n"
                "Enter your Mac password if prompted. When it finishes, click "
                "Check / Install again to verify."
            )
        else:
            import webbrowser
            webbrowser.open("https://existential.audio/blackhole/")
            self._bh.set_warn(
                "Homebrew not found — download page opened in your browser.\n"
                "Install manually, then click Check / Install again."
            )

    def _restart_coreaudio(self) -> None:
        """Restart the Core Audio server via a Terminal window (requires sudo)."""
        subprocess.run(
            [
                "osascript",
                "-e", 'tell application "Terminal" to activate',
                "-e", 'tell application "Terminal" to do script "sudo killall coreaudiod"',
            ],
            check=False,
        )
        self._bh.set_info(
            "A Terminal window has opened. Enter your Mac password to restart "
            "the audio server.\n"
            "Wait a couple of seconds, then click Check / Install again."
        )

    # ── Step 2: Multi-Output Device ───────────────────────────────────────────

    def _open_audio_midi(self) -> None:
        subprocess.run(["open", "-a", "Audio MIDI Setup"], check=False)

    # ── Step 3: AI Provider ───────────────────────────────────────────────────

    def _on_provider_change(self, provider: str) -> None:
        save_env_value("AI_PROVIDER", provider)
        config.ai_provider = provider
        self._update_provider_ui()
        self._check_provider()

    def _update_provider_ui(self) -> None:
        is_ollama = config.ai_provider == "ollama"
        self._api_key_widget.setVisible(not is_ollama)
        self._ollama_widget.setVisible(is_ollama)
        self._api.btn.setText("Check Connection" if is_ollama else "Check Key")

    def _check_provider(self) -> None:
        if config.ai_provider == "ollama":
            self._check_ollama()
        else:
            self._check_api_key()

    def _check_api_key(self) -> None:
        if _api_key_ok():
            self._api.set_ok(f"{config.ai_provider.capitalize()} API key is configured.")
            self._api_save_btn.setText("Replace Key")
        else:
            self._api.set_fail("No API key found — paste it below.")
            self._api_save_btn.setText("Save to .env")

    def _check_ollama(self) -> None:
        import urllib.request
        model = self._ollama_model_input.text().strip() or "llama3.1"

        # Phase 1 — quick server ping (synchronous, fast)
        try:
            urllib.request.urlopen(f"{config.ollama_base_url}/api/tags", timeout=3)
        except Exception:
            self._api.set_fail(
                f"Cannot reach Ollama at {config.ollama_base_url}.\n"
                "Install from ollama.com, then run:  ollama pull llama3.1"
            )
            return

        # Server is up — save model name, then test it in the background
        save_env_value("OLLAMA_MODEL", model)
        config.ollama_model = model
        self._api.btn.setEnabled(False)
        self._api.btn.setText("Testing model…")
        self._api.set_info(f"Ollama is running. Sending test prompt to {model}…")
        self._api_progress.setVisible(True)

        sigs = _OllamaSignals()
        sigs.ok.connect(self._on_ollama_ok)
        sigs.error.connect(self._on_ollama_error)
        threading.Thread(
            target=_run_ollama_test,
            args=(config.ollama_base_url, model, sigs),
            daemon=True,
        ).start()

    def _on_ollama_ok(self, reply: str) -> None:
        self._api_progress.setVisible(False)
        self._api.btn.setEnabled(True)
        self._api.btn.setText("Check Connection")
        preview = reply[:120] + ("…" if len(reply) > 120 else "")
        self._api.set_ok(
            f"Ollama + {config.ollama_model} is working. No data leaves this Mac.\n"
            f"Model said: \"{preview}\""
        )

    def _on_ollama_error(self, msg: str) -> None:
        self._api_progress.setVisible(False)
        self._api.btn.setEnabled(True)
        self._api.btn.setText("Check Connection")
        self._api.set_fail(
            f"Model '{config.ollama_model}' is not available.\n"
            f"Run:  ollama pull {config.ollama_model}\n"
            f"Error: {msg}"
        )

    def _save_api_key(self) -> None:
        key = self._api_key_input.text().strip()
        if not key:
            self._api.set_warn("Please paste a key before saving.")
            return
        key_name = (
            "ANTHROPIC_API_KEY" if config.ai_provider == "anthropic" else "OPENAI_API_KEY"
        )
        save_env_value(key_name, key)
        config.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY", "")
        config.openai_api_key = os.getenv("OPENAI_API_KEY", "")
        self._api_key_input.clear()
        self._check_api_key()

    # ── Step 4: spaCy Model ───────────────────────────────────────────────────

    def _check_spacy(self) -> None:
        if _spacy_model_installed():
            self._spacy.set_ok("en_core_web_lg is installed.")
        else:
            self._start_spacy_download()

    def _start_spacy_download(self) -> None:
        self._spacy.btn.setEnabled(False)
        self._spacy.btn.setText("Downloading…")
        self._spacy.set_info("Downloading en_core_web_lg (~560 MB)…")
        self._spacy_progress.setVisible(True)

        sigs = _SpacySignals()
        sigs.ok.connect(self._on_spacy_ok)
        sigs.error.connect(self._on_spacy_error)
        threading.Thread(target=_download_spacy_model, args=(sigs,), daemon=True).start()

    def _on_spacy_ok(self) -> None:
        self._spacy_progress.setVisible(False)
        self._spacy.btn.setEnabled(True)
        self._spacy.btn.setText("Check / Download")
        self._spacy.set_ok("en_core_web_lg downloaded and ready.")

    def _on_spacy_error(self, msg: str) -> None:
        self._spacy_progress.setVisible(False)
        self._spacy.btn.setEnabled(True)
        self._spacy.btn.setText("Check / Download")
        self._spacy.set_fail(f"Download failed: {msg}")

    # ── Step 5: Microphone Test ───────────────────────────────────────────────

    def _start_mic_test(self) -> None:
        self._mic_countdown = _MIC_RECORD_SECONDS
        self._mic.btn.setEnabled(False)
        self._mic.btn.setText(f"Recording...  {self._mic_countdown}s")
        self._mic.status.setText("⏺")
        self._mic.status.setStyleSheet("color: #e74c3c; font-size: 16px;")
        self._mic.result.setText("Read the script aloud now!")
        self._mic.result.setStyleSheet("color: white; font-size: 12px;")

        tmpdir = Path(tempfile.mkdtemp())
        self._mic_recorder = AudioRecorder(
            output_dir=tmpdir,
            primary_device=config.audio_primary_device,
        )
        self._mic_recorder.start()
        self._mic_timer.start()

    def _mic_tick(self) -> None:
        self._mic_countdown -= 1
        if self._mic_countdown > 0:
            self._mic.btn.setText(f"Recording...  {self._mic_countdown}s")
            return

        self._mic_timer.stop()
        recording = self._mic_recorder.stop()
        audio_path = recording.therapist_path
        self._mic.btn.setText("Transcribing...")
        self._mic.status.setText("⏳")
        self._mic.status.setStyleSheet("color: orange; font-size: 16px;")
        self._mic.result.setText(
            "Transcribing — if this is the first run, the Whisper model is "
            "downloading (~250 MB). This will be fast on future runs."
        )
        self._mic.result.setStyleSheet("color: gray; font-size: 12px;")

        signals = _TestSignals()
        signals.result.connect(self._on_mic_result)
        signals.error.connect(lambda m: self._on_test_error(self._mic, m))
        threading.Thread(
            target=_transcribe_and_compare,
            args=(audio_path, _MIC_SCRIPT, signals),
            daemon=True,
        ).start()

    def _on_mic_result(self, transcript: str, ratio: float) -> None:
        self._mic.btn.setEnabled(True)
        self._mic.btn.setText(f"Record & Test ({_MIC_RECORD_SECONDS}s)")
        _apply_result(self._mic, transcript, ratio)

    # ── Step 5: Loopback Test ─────────────────────────────────────────────────

    def _save_loopback_device(self) -> None:
        """Write AUDIO_LOOPBACK_DEVICE=BlackHole 2ch to .env and reload config."""
        save_env_value("AUDIO_LOOPBACK_DEVICE", "BlackHole 2ch")
        config.audio_loopback_device = "BlackHole 2ch"

    def _run_loopback_test(self) -> None:
        self._save_loopback_device()
        self._lb.btn.setEnabled(False)
        self._lb.btn.setText("Running test...")
        self._lb.status.setText("⏳")
        self._lb.status.setStyleSheet("color: orange; font-size: 16px;")
        self._lb.result.setText("Playing tone to Therapy Output and recording from BlackHole...")
        self._lb.result.setStyleSheet("color: gray; font-size: 12px;")

        signals = _TestSignals()
        signals.result.connect(self._on_loopback_result)
        signals.error.connect(lambda m: self._on_test_error(self._lb, m))
        threading.Thread(
            target=_run_loopback_capture,
            args=(signals,),
            daemon=True,
        ).start()

    def _on_loopback_result(self, transcript: str, ratio: float) -> None:
        self._lb.btn.setEnabled(True)
        self._lb.btn.setText("Run Test")
        _apply_result(self._lb, transcript, ratio)

    # ── Update App ────────────────────────────────────────────────────────────

    def _run_update(self) -> None:
        self._update.btn.setEnabled(False)
        self._update.btn.setText("Updating…")
        self._update.set_info("Running git pull…")
        self._update_progress.setVisible(True)

        sigs = _UpdateSignals()
        sigs.ok.connect(self._on_update_ok)
        sigs.error.connect(self._on_update_error)
        threading.Thread(target=_do_update, args=(sigs,), daemon=True).start()

    def _on_update_ok(self, msg: str) -> None:
        self._update_progress.setVisible(False)
        self._update.btn.setEnabled(True)
        self._update.btn.setText("Check for Updates")
        self._update.set_ok(msg)

    def _on_update_error(self, msg: str) -> None:
        self._update_progress.setVisible(False)
        self._update.btn.setEnabled(True)
        self._update.btn.setText("Check for Updates")
        self._update.set_fail(msg)

    # ── Shared error handler ──────────────────────────────────────────────────

    def _on_test_error(self, step: _Step, msg: str) -> None:
        step.btn.setEnabled(True)
        if step is self._mic:
            step.btn.setText(f"Record & Test ({_MIC_RECORD_SECONDS}s)")
        elif step is self._lb:
            step.btn.setText("Run Test")
        step.set_fail(f"Error: {msg}")


# ── Pure functions run in background threads ──────────────────────────────────

def _transcribe_and_compare(
    audio_path: Path, expected: str, signals: _TestSignals
) -> None:
    try:
        transcript = Transcriber(config.whisper_model).transcribe(audio_path)
        ratio = _word_match(expected, transcript)
        signals.result.emit(transcript.strip(), ratio)
    except Exception as exc:
        signals.error.emit(str(exc))


def _run_loopback_capture(signals: _TestSignals) -> None:
    """Play a 440 Hz tone to 'Therapy Output' and record from BlackHole 2ch."""
    try:
        import sounddevice as sd
        import numpy as np
        from therapy_notes.audio.devices import find_device

        SAMPLE_RATE = 16_000
        DURATION = 3
        TONE_HZ = 440
        SIGNAL_THRESHOLD = 0.005

        # Locate required devices
        blackhole_idx = find_device("BlackHole 2ch", kind="input")
        therapy_idx = find_device("Therapy Output", kind="output")

        if blackhole_idx is None:
            signals.error.emit(
                "BlackHole 2ch input device not found. "
                "Complete Step 1 and restart Core Audio."
            )
            return
        if therapy_idx is None:
            signals.error.emit(
                "'Therapy Output' device not found. "
                "Create the Multi-Output Device in Step 2 first."
            )
            return

        # Generate tone
        t = np.linspace(0, DURATION, int(SAMPLE_RATE * DURATION), endpoint=False)
        tone = (0.4 * np.sin(2 * np.pi * TONE_HZ * t)).astype(np.float32)

        recorded: list = []

        def _cb(indata, frames, time_info, status):
            recorded.append(indata.copy())

        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            device=blackhole_idx,
            callback=_cb,
            dtype="float32",
        ):
            sd.play(tone, samplerate=SAMPLE_RATE, device=therapy_idx, blocking=True)
            time.sleep(0.3)

        if not recorded:
            signals.result.emit("Nothing captured from BlackHole 2ch.", 0.0)
            return

        captured = np.concatenate(recorded).flatten()
        peak = float(np.max(np.abs(captured)))

        if peak >= SIGNAL_THRESHOLD:
            signals.result.emit(f"Tone captured cleanly  (peak={peak:.3f})", 1.0)
        else:
            signals.result.emit(
                f"Signal too weak  (peak={peak:.4f})\n"
                "Make sure 'Therapy Output' is selected as your system speaker output "
                "and that BlackHole 2ch is checked in the Multi-Output Device.",
                0.0,
            )
    except Exception as exc:
        signals.error.emit(str(exc))


def _run_ollama_test(base_url: str, model: str, signals: _OllamaSignals) -> None:
    """Send a short test prompt to Ollama and emit the reply."""
    try:
        import openai as _openai
        client = _openai.OpenAI(base_url=f"{base_url}/v1", api_key="ollama")
        response = client.chat.completions.create(
            model=model,
            messages=[{
                "role": "user",
                "content": (
                    "Reply with one sentence only: confirm you are ready "
                    "to assist with clinical documentation."
                ),
            }],
            max_tokens=60,
        )
        reply = response.choices[0].message.content.strip()
        signals.ok.emit(reply)
    except Exception as exc:
        signals.error.emit(str(exc))


def _do_update(signals: _UpdateSignals) -> None:
    try:
        repo_root = Path(__file__).parents[3]

        pull = subprocess.run(
            ["git", "pull"],
            capture_output=True,
            text=True,
            cwd=repo_root,
        )
        if pull.returncode != 0:
            signals.error.emit(pull.stderr.strip() or pull.stdout.strip())
            return

        already_up_to_date = "Already up to date." in pull.stdout

        if not already_up_to_date:
            sync = subprocess.run(
                ["uv", "sync"],
                capture_output=True,
                text=True,
                cwd=repo_root,
            )
            if sync.returncode != 0:
                signals.error.emit(sync.stderr.strip() or sync.stdout.strip())
                return
            signals.ok.emit("Updated successfully — please restart the app.")
        else:
            signals.ok.emit("Already up to date.")
    except Exception as exc:
        signals.error.emit(str(exc))


def _download_spacy_model(signals: _SpacySignals) -> None:
    try:
        result = subprocess.run(
            [sys.executable, "-m", "spacy", "download", "en_core_web_lg"],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            signals.ok.emit()
        else:
            signals.error.emit(result.stderr.strip() or result.stdout.strip())
    except Exception as exc:
        signals.error.emit(str(exc))


def _apply_result(step: "SetupWindow._Step", transcript: str, ratio: float) -> None:
    pct = int(ratio * 100)
    passed = ratio >= _MATCH_THRESHOLD
    label = "Pass" if passed else "Low match — check setup"
    if passed:
        step.set_ok(f"{label}  ({pct}% word match)\nHeard: \"{transcript}\"")
    else:
        step.set_warn(f"{label}  ({pct}% word match)\nHeard: \"{transcript}\"")
