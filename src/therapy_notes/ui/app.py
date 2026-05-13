from __future__ import annotations

import subprocess
import sys
import threading
from pathlib import Path

from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QObject
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from ..audio.recorder import AudioRecorder, RecordingResult
from ..anonymization.anonymizer import Anonymizer
from ..config import config
from ..notes.generator import NoteGenerator
from ..transcription.transcriber import Transcriber
from .setup_window import SetupWindow

_BTN_START = """
    QPushButton {
        background-color: #1a73e8;
        color: white;
        border-radius: 6px;
        font-size: 14px;
        font-weight: bold;
        padding: 6px 0;
    }
    QPushButton:hover { background-color: #1557b0; }
    QPushButton:disabled { background-color: #888; color: #ccc; }
"""
_BTN_STOP = """
    QPushButton {
        background-color: #c0392b;
        color: white;
        border-radius: 6px;
        font-size: 14px;
        font-weight: bold;
        padding: 6px 0;
    }
    QPushButton:hover { background-color: #922b21; }
"""


class _Signals(QObject):
    """Carries cross-thread signals from the pipeline worker back to the UI."""
    status = pyqtSignal(str)
    finished = pyqtSignal(str, str)   # notes text, notes file path
    error = pyqtSignal(str)


class TherapyNotesApp(QWidget):
    def __init__(self) -> None:
        super().__init__()

        self._recorder = AudioRecorder(
            output_dir=config.sessions_dir,
            primary_device=config.audio_primary_device,
            loopback_device=config.audio_loopback_device,
        )
        self._total_seconds = config.session_duration_minutes * 60
        self._elapsed = 0
        self._running = False

        self._tick_timer = QTimer()
        self._tick_timer.setInterval(1000)
        self._tick_timer.timeout.connect(self._tick)

        self._build_ui()
        self._check_config()

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        self.setWindowTitle("Therapy Notes")
        self.setWindowFlags(
            Qt.WindowType.WindowStaysOnTopHint |
            Qt.WindowType.Window
        )
        self.setFixedSize(420, 195)

        # Top-center of primary screen
        screen = QApplication.primaryScreen().geometry()
        self.move((screen.width() - 420) // 2, 12)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 10, 16, 12)
        layout.setSpacing(4)

        self._status_lbl = QLabel("Ready")
        self._status_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._status_lbl.setStyleSheet("color: gray; font-size: 12px;")
        layout.addWidget(self._status_lbl)

        self._timer_lbl = QLabel(self._fmt(0))
        self._timer_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._timer_lbl.setFont(QFont("Courier New", 20, QFont.Weight.Bold))
        layout.addWidget(self._timer_lbl)

        self._progress = QProgressBar()
        self._progress.setRange(0, 0)   # indeterminate spinner
        self._progress.setFixedHeight(5)
        self._progress.setVisible(False)
        layout.addWidget(self._progress)

        self._btn = QPushButton("Start Meeting")
        self._btn.setStyleSheet(_BTN_START)
        self._btn.clicked.connect(self._on_button)
        layout.addWidget(self._btn)

        self._couples_cb = QCheckBox("Couples session")
        self._couples_cb.setStyleSheet("color: gray; font-size: 11px;")
        layout.addWidget(self._couples_cb, alignment=Qt.AlignmentFlag.AlignHCenter)

        self._setup_btn = QPushButton("Setup & Test")
        self._setup_btn.setStyleSheet(
            "QPushButton { color: gray; background: transparent; "
            "border: none; font-size: 11px; }"
            "QPushButton:hover { color: white; }"
        )
        self._setup_btn.clicked.connect(self._open_setup)
        layout.addWidget(self._setup_btn)

    # ── Config check ──────────────────────────────────────────────────────────

    def _check_config(self) -> None:
        errors = config.validate()
        if errors:
            self._status_lbl.setText("Setup required — click Setup & Test")
            self._status_lbl.setStyleSheet("color: orange; font-size: 12px;")
            self._btn.setEnabled(False)
            for e in errors:
                print(f"[config] {e}")
            # Auto-open setup on first launch when not configured
            QTimer.singleShot(300, self._open_setup)

    def _open_setup(self) -> None:
        win = SetupWindow(self)
        win.exec()

    # ── Button ────────────────────────────────────────────────────────────────

    def _on_button(self) -> None:
        if not self._running:
            self._start_session()
        else:
            self._stop_session()

    # ── Session lifecycle ─────────────────────────────────────────────────────

    def _start_session(self) -> None:
        self._running = True
        self._elapsed = 0
        self._recorder.start()
        self._btn.setText("Stop Meeting")
        self._btn.setStyleSheet(_BTN_STOP)
        self._couples_cb.setEnabled(False)
        self._status_lbl.setText("Recording...")
        self._status_lbl.setStyleSheet("color: white; font-size: 12px;")
        self._tick_timer.start()

    def _stop_session(self) -> None:
        self._running = False
        self._tick_timer.stop()
        recording = self._recorder.stop()
        self._btn.setEnabled(False)
        self._btn.setText("Processing...")
        self._progress.setVisible(True)

        couples = self._couples_cb.isChecked()
        signals = _Signals()
        signals.status.connect(self._on_status)
        signals.finished.connect(self._on_done)
        signals.error.connect(self._on_error)
        threading.Thread(
            target=self._run_pipeline,
            args=(recording, signals, couples),
            daemon=True,
        ).start()

    def _tick(self) -> None:
        self._elapsed += 1
        self._timer_lbl.setText(self._fmt(self._elapsed))
        if self._elapsed >= self._total_seconds:
            self._stop_session()

    # ── Processing pipeline (background thread) ───────────────────────────────

    def _run_pipeline(self, recording: RecordingResult, signals: _Signals, couples: bool = False) -> None:
        session_dir = recording.session_dir
        try:
            t = Transcriber(config.whisper_model)
            if recording.client_path is not None:
                signals.status.emit("Transcribing audio (therapist + client)...")
                transcript = t.transcribe_two_speakers(
                    recording.therapist_path, recording.client_path
                )
            else:
                signals.status.emit("Transcribing audio — this may take a few minutes...")
                transcript = t.transcribe(recording.therapist_path)

            raw_path = session_dir / "transcript_raw.txt"
            raw_path.write_text(transcript, encoding="utf-8")

            local_only = config.ai_provider == "ollama"
            if local_only:
                notes_input = transcript
            else:
                signals.status.emit("Anonymizing transcript...")
                notes_input = Anonymizer().anonymize(transcript)
                (session_dir / "transcript_anon.txt").write_text(notes_input, encoding="utf-8")

            if config.delete_raw_transcript:
                raw_path.unlink(missing_ok=True)
            if config.delete_audio_after_transcription:
                recording.therapist_path.unlink(missing_ok=True)
                if recording.client_path:
                    recording.client_path.unlink(missing_ok=True)

            signals.status.emit("Generating SOAP notes...")
            notes = NoteGenerator(couples=couples, anonymized=not local_only).generate(notes_input)
            notes_path = session_dir / "notes.txt"
            notes_path.write_text(notes, encoding="utf-8")

            signals.finished.emit(notes, str(notes_path))

        except Exception as exc:  # noqa: BLE001
            signals.error.emit(str(exc))

    # ── Slot handlers (main thread) ───────────────────────────────────────────

    def _on_status(self, text: str) -> None:
        self._status_lbl.setText(text)

    def _on_done(self, notes: str, notes_path_str: str) -> None:
        notes_path = Path(notes_path_str)
        self._progress.setVisible(False)
        self._btn.setEnabled(True)
        self._btn.setText("Start Meeting")
        self._btn.setStyleSheet(_BTN_START)
        self._couples_cb.setEnabled(True)
        self._timer_lbl.setText(self._fmt(0))
        self._status_lbl.setText(f"Done — saved to {notes_path.parent.name}/")
        self._status_lbl.setStyleSheet("color: gray; font-size: 12px;")
        self._open_notes_viewer(notes, notes_path)

    def _on_error(self, message: str) -> None:
        self._progress.setVisible(False)
        self._btn.setEnabled(True)
        self._btn.setText("Start Meeting")
        self._btn.setStyleSheet(_BTN_START)
        self._couples_cb.setEnabled(True)
        self._status_lbl.setText(f"Error: {message}")
        self._status_lbl.setStyleSheet("color: #e74c3c; font-size: 12px;")
        print(f"[error] {message}")

    # ── Notes viewer ──────────────────────────────────────────────────────────

    def _open_notes_viewer(self, notes: str, notes_path: Path) -> None:
        dialog = QDialog(self)
        dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        dialog.setWindowTitle("Session Notes")
        dialog.resize(740, 680)

        layout = QVBoxLayout(dialog)

        text = QTextEdit()
        text.setPlainText(notes)
        text.setFont(QFont("Helvetica", 13))
        layout.addWidget(text)

        buttons = QHBoxLayout()

        copy_btn = QPushButton("Copy to Clipboard")
        copy_btn.clicked.connect(lambda: QApplication.clipboard().setText(notes))
        buttons.addWidget(copy_btn)

        folder_btn = QPushButton("Open Session Folder")
        folder_btn.clicked.connect(
            lambda: subprocess.run(["open", str(notes_path.parent)], check=False)
        )
        buttons.addWidget(folder_btn)
        buttons.addStretch()

        layout.addLayout(buttons)
        dialog.show()

    # ── Helper ────────────────────────────────────────────────────────────────

    def _fmt(self, elapsed: int) -> str:
        em, es = divmod(elapsed, 60)
        tm, ts = divmod(self._total_seconds, 60)
        return f"{em:02d}:{es:02d}  /  {tm:02d}:{ts:02d}"


def run() -> None:
    app = QApplication(sys.argv)
    window = TherapyNotesApp()
    window.show()
    sys.exit(app.exec())
