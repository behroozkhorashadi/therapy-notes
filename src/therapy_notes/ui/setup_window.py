from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
import threading
import webbrowser
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

from ..config import config, reload_config
from ..models import recommend_models
from ..setup_tasks import MIC_RECORD_SECONDS as _MIC_RECORD_SECONDS
from ..setup_tasks import MIC_SCRIPT as _MIC_SCRIPT
from ..setup_tasks import word_match as _word_match  # noqa: F401 - used by tests
from ..utils.env import save_env_value
from .processes import module_command, module_env

# ── Test constants ────────────────────────────────────────────────────────────

_MATCH_THRESHOLD = 0.50   # 50% word overlap = pass


# ── Helpers ───────────────────────────────────────────────────────────────────

def _blackhole_installed() -> bool:
    """True if the BlackHole HAL driver file exists on disk (may not be loaded yet)."""
    return Path("/Library/Audio/Plug-Ins/HAL/BlackHole2ch.driver").exists()


def _homebrew_available() -> bool:
    return shutil.which("brew") is not None


def _api_key_ok() -> bool:
    return not config.validate() or not any("API_KEY" in error for error in config.validate())


def _spacy_model_installed() -> bool:
    return importlib.util.find_spec("en_core_web_lg") is not None


def _run_task(task: str, timeout: float) -> dict:
    """Run a setup task in its own process. Its result is the last line of output."""
    try:
        result = subprocess.run(
            module_command("setup_tasks", task), capture_output=True, text=True,
            timeout=timeout, env=module_env(),
        )
    except subprocess.TimeoutExpired:
        return {"error": "The test timed out. Audio devices may not be responding."}
    lines = result.stdout.strip().splitlines()
    try:
        return json.loads(lines[-1])
    except (IndexError, ValueError):
        return {"error": (result.stderr.strip() or "The test failed.")[-300:]}


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


class _DeviceSignals(QObject):
    result = pyqtSignal(bool)   # BlackHole visible to Core Audio


class _PingSignals(QObject):
    done = pyqtSignal(bool)


class _UpdateSignals(QObject):
    ok    = pyqtSignal(str)   # summary of what changed
    error = pyqtSignal(str)



# ── Setup window ──────────────────────────────────────────────────────────────

class SetupWindow(QDialog):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Therapy Notes — Setup & Test")
        self.setMinimumSize(820, 560)
        self.resize(820, 720)

        self._mic_countdown = _MIC_RECORD_SECONDS
        self._mic_timer = QTimer(self)
        self._mic_timer.setInterval(1000)
        self._mic_timer.timeout.connect(self._mic_tick)

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
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
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
        self._ollama_model_input.setPlaceholderText(recommend_models().local_llm or "qwen3.5:9b")
        self._ollama_model_input.setText(config.ollama_model)
        ollama_row.addWidget(self._ollama_model_input)
        ollama_row.addStretch()
        self._ollama_widget = QWidget()
        self._ollama_widget.setLayout(ollama_row)
        self._api.col.addWidget(self._ollama_widget)

        # Indeterminate progress bar — visible only while Ollama model test runs
        self._api_progress = QProgressBar()
        self._api_progress.setRange(0, 0)
        self._api_progress.setFixedHeight(4)
        self._api_progress.setTextVisible(False)
        self._api_progress.setVisible(False)
        self._api.col.addWidget(self._api_progress)

        # Set the initial state before wiring, so opening the window changes nothing.
        p = config.ai_provider
        if p == "openai":
            self._rb_openai.setChecked(True)
        elif p == "ollama":
            self._rb_ollama.setChecked(True)
        else:
            self._rb_anthropic.setChecked(True)
        self._update_provider_ui()  # show/hide correct input row
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

        rec = recommend_models()
        rec_label = QLabel(f"Recommended here: {rec.summary}")
        rec_label.setWordWrap(True)
        rec_label.setStyleSheet("color: gray; font-size: 11px;")
        self._api.col.addWidget(rec_label)

        # ── Step 4: spaCy Language Model ──────────────────────────────────────
        self._spacy = self._step(
            "Step 4 — Anonymization Model",
            "The English NLP model (en_core_web_lg) that Presidio uses to redact "
            "names, locations, and other identifying details from transcripts "
            "before they go to a cloud provider. Normally installed with the app.",
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
            "The speech model will transcribe it and check how well it matches.\n\n"
            "⚠ First run only: the speech model downloads before transcription "
            "begins. This is a one-time step.",
            btn=f"Record && Test ({_MIC_RECORD_SECONDS}s)",
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
            "Zoom cannot be tested automatically, but Step 6 validates the same "
            "audio path. If Step 6 passes, you know:\n"
            "  • BlackHole is installed and receiving audio\n"
            "  • Your speakers are routing through BlackHole\n\n"
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
        """Quick, read-only checks on open. Nothing downloads, loads or writes."""
        self._refresh_blackhole()
        if config.ai_provider == "ollama":
            self._ping_ollama()
        else:
            self._check_api_key()
        if _spacy_model_installed():
            self._spacy.set_ok("en_core_web_lg is installed.")
        else:
            self._spacy.set_warn("Not installed yet. Click Check / Download (~560 MB).")

    # ── Step 1: BlackHole ─────────────────────────────────────────────────────

    def _refresh_blackhole(self) -> None:
        signals = _DeviceSignals(self)
        signals.result.connect(self._on_blackhole_checked)

        def task() -> None:
            result = _run_task("devices", timeout=20)
            signals.result.emit(bool(result.get("blackhole")))

        threading.Thread(target=task, daemon=True).start()

    def _on_blackhole_checked(self, loaded: bool) -> None:
        if loaded:
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
            self._refresh_blackhole()
            return

        if _homebrew_available():
            # brew install requires sudo for the .pkg step, which needs an
            # interactive terminal to accept a password. Open Terminal so the
            # user can see the prompt and type their password normally.
            subprocess.Popen([
                "osascript",
                "-e", 'tell application "Terminal" to activate',
                "-e", 'tell application "Terminal" to do script "brew install blackhole-2ch"',
            ])
            self._bh.status.setText("⏳")
            self._bh.status.setStyleSheet("color: orange;")
            self._bh.set_info(
                "A Terminal window has opened running:\n"
                "    brew install blackhole-2ch\n\n"
                "Enter your Mac password if prompted. When it finishes, click "
                "Check / Install again to verify."
            )
        else:
            webbrowser.open("https://existential.audio/blackhole/")
            self._bh.set_warn(
                "Homebrew not found — download page opened in your browser.\n"
                "Install manually, then click Check / Install again."
            )

    def _restart_coreaudio(self) -> None:
        """Restart the Core Audio server via a Terminal window (requires sudo)."""
        subprocess.Popen([
            "osascript",
            "-e", 'tell application "Terminal" to activate',
            "-e", 'tell application "Terminal" to do script "sudo killall coreaudiod"',
        ])
        self._bh.set_info(
            "A Terminal window has opened. Enter your Mac password to restart "
            "the audio server.\n"
            "Wait a couple of seconds, then click Check / Install again."
        )

    # ── Step 2: Multi-Output Device ───────────────────────────────────────────

    def _open_audio_midi(self) -> None:
        subprocess.Popen(["open", "-a", "Audio MIDI Setup"])

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

    def _ping_ollama(self) -> None:
        """Quick reachability check, off the UI thread. Doesn't load a model."""
        url = config.ollama_base_url
        signals = _PingSignals(self)
        signals.done.connect(self._on_ollama_ping)

        def task() -> None:
            import urllib.request
            try:
                urllib.request.urlopen(f"{url}/api/tags", timeout=3)
                signals.done.emit(True)
            except OSError:
                signals.done.emit(False)

        threading.Thread(target=task, daemon=True).start()

    def _on_ollama_ping(self, reachable: bool) -> None:
        if reachable:
            self._api.set_info("Ollama is running. Click Check Connection to test the model.")
        else:
            self._api.set_fail(
                f"Cannot reach Ollama at {config.ollama_base_url}.\n"
                f"Install from ollama.com, then run:  ollama pull {config.ollama_model}"
            )

    def _check_ollama(self) -> None:
        """Send a real test prompt. Loads the model, so only runs when asked."""
        model = self._ollama_model_input.text().strip() or config.ollama_model
        save_env_value("OLLAMA_MODEL", model)
        config.ollama_model = model
        self._api.btn.setEnabled(False)
        self._api.btn.setText("Testing model…")
        self._api.set_info(f"Sending a test prompt to {model}…")
        self._api_progress.setVisible(True)

        sigs = _OllamaSignals(self)
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
        reload_config()
        self._api_key_input.clear()
        self._check_api_key()

    # ── Step 4: spaCy Model ───────────────────────────────────────────────────

    def _check_spacy(self) -> None:
        """Runs on click only, so a large download never starts by surprise."""
        if _spacy_model_installed():
            self._spacy.set_ok("en_core_web_lg is installed.")
        else:
            self._start_spacy_download()

    def _start_spacy_download(self) -> None:
        self._spacy.btn.setEnabled(False)
        self._spacy.btn.setText("Downloading…")
        self._spacy.set_info("Downloading en_core_web_lg (~560 MB)…")
        self._spacy_progress.setVisible(True)

        sigs = _SpacySignals(self)
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
        self._mic.result.setStyleSheet("font-size: 12px;")
        self._mic_timer.start()

        signals = _TestSignals(self)
        signals.result.connect(self._on_mic_result)
        signals.error.connect(lambda m: self._on_test_error(self._mic, m))
        # First run may download the speech model, so allow plenty of time.
        threading.Thread(target=_run_test_task, args=("mic-test", 1800, signals), daemon=True).start()

    def _mic_tick(self) -> None:
        self._mic_countdown -= 1
        if self._mic_countdown > 0:
            self._mic.btn.setText(f"Recording...  {self._mic_countdown}s")
            return
        self._mic_timer.stop()
        self._mic.btn.setText("Transcribing...")
        self._mic.status.setText("⏳")
        self._mic.status.setStyleSheet("color: orange; font-size: 16px;")
        self._mic.result.setText(
            "Transcribing — on the first run, the speech model downloads first. "
            "Future runs will be fast."
        )
        self._mic.result.setStyleSheet("color: gray; font-size: 12px;")

    def _on_mic_result(self, transcript: str, ratio: float) -> None:
        self._mic_timer.stop()
        self._mic.btn.setEnabled(True)
        self._mic.btn.setText(f"Record && Test ({_MIC_RECORD_SECONDS}s)")
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

        signals = _TestSignals(self)
        signals.result.connect(self._on_loopback_result)
        signals.error.connect(lambda m: self._on_test_error(self._lb, m))
        threading.Thread(target=_run_test_task, args=("loopback-test", 60, signals), daemon=True).start()

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

        sigs = _UpdateSignals(self)
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
        self._mic_timer.stop()
        step.btn.setEnabled(True)
        if step is self._mic:
            step.btn.setText(f"Record && Test ({_MIC_RECORD_SECONDS}s)")
        elif step is self._lb:
            step.btn.setText("Run Test")
        step.set_fail(f"Error: {msg}")


# ── Pure functions run in background threads ──────────────────────────────────

def _run_test_task(task: str, timeout: float, signals: _TestSignals) -> None:
    result = _run_task(task, timeout)
    if "error" in result:
        signals.error.emit(str(result["error"]))
    else:
        signals.result.emit(result.get("transcript", ""), float(result.get("ratio", 0.0)))


def _run_ollama_test(base_url: str, model: str, signals: _OllamaSignals) -> None:
    """Send a short test prompt to Ollama and emit the reply."""
    import urllib.request
    try:
        body = {
            "model": model, "stream": False, "options": {"num_predict": 60},
            "messages": [{"role": "user", "content": "Reply with one sentence confirming you are ready."}],
        }
        request = urllib.request.Request(
            f"{base_url}/api/chat", data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=300) as response:
            signals.ok.emit(json.load(response)["message"]["content"].strip())
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
            timeout=120,
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
                timeout=900,
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
