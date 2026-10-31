"""Main window. The UI only displays state and sends commands.

Recording and processing run in separate processes (recorder_service, worker) and
sessions live on disk, so a crash in either can't freeze this window or lose work.
"""
from __future__ import annotations

import json
import logging
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from PyQt6.QtCore import QLockFile, QObject, QProcess, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from .. import session as store
from ..config import config, data_dir, reload_config
from ..logging_setup import setup_logging
from .processes import make_process
from .setup_window import SetupWindow

log = logging.getLogger("ui")

_BUTTON = """
    QPushButton {{
        background-color: {color}; color: white; border-radius: 6px;
        font-size: 14px; font-weight: bold; padding: 6px 0;
    }}
    QPushButton:hover {{ background-color: {hover}; }}
    QPushButton:disabled {{ background-color: #888; color: #ccc; }}
"""
_BTN_START = _BUTTON.format(color="#1a73e8", hover="#1557b0")
_BTN_STOP = _BUTTON.format(color="#c0392b", hover="#922b21")

_STOP_TIMEOUT_MS = 15_000
_START_TIMEOUT_MS = 20_000
_WORKER_RESTART_DELAY = 30.0
_SESSION_LIMIT = 50

_STATE_TEXT = {
    store.RECORDING: "Recording",
    store.QUEUED: "Waiting to process",
    store.TRANSCRIBING: "Transcribing",
    store.ANONYMIZING: "Removing identifying details",
    store.GENERATING: "Writing notes",
    store.DONE: "Notes ready",
    store.FAILED: "Failed",
    store.UNPROCESSED: "Audio saved, no notes",
}


def session_label(data: dict) -> str:
    try:
        when = datetime.strptime(data["id"][:19], "%Y-%m-%d_%H-%M-%S").strftime("%b %-d, %-I:%M %p")
    except ValueError:
        when = data["id"]
    label = f"{when} — {_STATE_TEXT.get(data.get('state'), data.get('state', 'Unknown'))}"
    if data.get("warnings"):
        label += " ⚠"
    return label


class RecorderClient(QObject):
    """Runs the recorder process and turns its output into signals."""

    started = pyqtSignal(str)
    status = pyqtSignal(dict)
    warning = pyqtSignal(str)
    stopped = pyqtSignal(bool)   # True if a recording was saved
    failed = pyqtSignal(str)

    def __init__(self, parent: QObject) -> None:
        super().__init__(parent)
        self._process: Optional[QProcess] = None
        self._buffer = b""
        self._running = False
        self._finished = False
        self._stopping = False
        self._saved = False
        self._start_timer = QTimer(self, singleShot=True, interval=_START_TIMEOUT_MS)
        self._start_timer.timeout.connect(self._start_timed_out)
        self._stop_timer = QTimer(self, singleShot=True, interval=_STOP_TIMEOUT_MS)
        self._stop_timer.timeout.connect(self._kill)

    def start(self, couples: bool) -> None:
        args = ["--couples"] if couples else []
        self._process = make_process(self, "recorder_service", *args)
        self._process.setProcessChannelMode(QProcess.ProcessChannelMode.ForwardedErrorChannel)
        self._process.readyReadStandardOutput.connect(self._read)
        self._process.finished.connect(self._on_finished)
        self._process.errorOccurred.connect(self._on_process_error)
        self._buffer = b""
        self._running = self._finished = self._stopping = self._saved = False
        self._start_timer.start()
        self._process.start()

    def stop(self) -> None:
        if self._process is None or self._stopping:
            return
        self._stopping = True
        self._process.write(b"stop\n")
        self._stop_timer.start()

    def _kill(self) -> None:
        # The audio on disk is safe; the worker repairs and queues it.
        log.warning("Recorder did not stop in time; killing it")
        if self._process is not None:
            self._process.kill()

    def _start_timed_out(self) -> None:
        log.error("Recorder did not start in time")
        self.failed.emit("The recorder didn't start. Check the audio devices in Setup, then try again.")
        self._finished = True
        self._kill()

    def _read(self) -> None:
        self._buffer += bytes(self._process.readAllStandardOutput())
        *lines, self._buffer = self._buffer.split(b"\n")
        for line in lines:
            try:
                message = json.loads(line)
            except ValueError:
                continue
            self._handle(message)

    def _handle(self, message: dict) -> None:
        event = message.get("event")
        if event == "started":
            self._start_timer.stop()
            self._running = True
            self.started.emit(message["session"])
        elif event == "status":
            self.status.emit(message)
        elif event == "warning":
            self.warning.emit(message.get("message", ""))
        elif event == "stopped":
            self._saved = not message.get("discarded", False)
        elif event == "error":
            self._start_timer.stop()
            self._finished = True
            self.failed.emit(message.get("message", "Could not start recording."))

    def _on_process_error(self, error: QProcess.ProcessError) -> None:
        if error == QProcess.ProcessError.FailedToStart:
            self._start_timer.stop()
            self._finished = True
            self.failed.emit("The recorder couldn't start.")

    def _on_finished(self, code: int, _status) -> None:
        self._start_timer.stop()
        self._stop_timer.stop()
        was_running, already_reported = self._running, self._finished
        expected = self._stopping
        self._running, self._finished = False, True
        self._process.deleteLater()
        self._process = None
        if already_reported:
            return
        if was_running and not expected:
            log.error("Recorder exited unexpectedly (code %s)", code)
            self.warning.emit("Recording stopped unexpectedly. The audio up to that point is saved.")
        self.stopped.emit(self._saved or was_running)


class TherapyNotesApp(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self._recorder = RecorderClient(self)
        self._recorder.started.connect(self._on_recording_started)
        self._recorder.status.connect(self._on_recording_status)
        self._recorder.warning.connect(self._show_warning)
        self._recorder.stopped.connect(self._on_recording_stopped)
        self._recorder.failed.connect(self._on_recording_failed)

        self._recording = False
        self._busy = False
        self._quit_after_stop = False
        self._started_at = 0.0
        self._limit_warned = False
        self._worker: Optional[QProcess] = None
        self._worker_failed_at = 0.0
        self._health: Optional[QProcess] = None
        self._checks: list[dict] = []
        self._states: dict[str, str] = {}
        self._selected_id: Optional[str] = None
        self._viewers: list[QDialog] = []

        self._tick_timer = QTimer(self, interval=500)
        self._tick_timer.timeout.connect(self._tick)
        self._poll_timer = QTimer(self, interval=3000)
        self._poll_timer.timeout.connect(self._refresh)
        self._health_timer = QTimer(self, interval=3_600_000)
        self._health_timer.timeout.connect(self._run_health)

        self._build_ui()
        self._check_config()
        self._refresh()
        self._poll_timer.start()
        self._health_timer.start()
        self._run_health()

    # ── UI ────────────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        self.setWindowTitle("Therapy Notes")
        self.setWindowFlags(Qt.WindowType.WindowStaysOnTopHint | Qt.WindowType.Window)
        self.setFixedWidth(420)
        screen = QApplication.primaryScreen().geometry()
        self.move((screen.width() - 420) // 2, 12)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 10, 16, 12)
        layout.setSpacing(4)
        layout.setSizeConstraint(QVBoxLayout.SizeConstraint.SetFixedSize)

        self._banner = QWidget()
        banner_row = QHBoxLayout(self._banner)
        banner_row.setContentsMargins(0, 0, 0, 4)
        self._banner_lbl = QLabel()
        self._banner_lbl.setWordWrap(True)
        self._banner_lbl.setFixedWidth(290)
        banner_row.addWidget(self._banner_lbl, 1)
        self._banner_btn = QPushButton("Details")
        self._banner_btn.clicked.connect(self._open_health)
        banner_row.addWidget(self._banner_btn)
        self._banner.setVisible(False)
        layout.addWidget(self._banner)

        self._status_lbl = QLabel("Ready")
        self._status_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._status_lbl.setWordWrap(True)
        self._status_lbl.setFixedWidth(388)
        self._set_status("Ready")
        layout.addWidget(self._status_lbl)

        self._timer_lbl = QLabel(self._fmt(0))
        self._timer_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._timer_lbl.setFont(QFont("Menlo", 20, QFont.Weight.Bold))
        layout.addWidget(self._timer_lbl)

        self._progress = QProgressBar()
        self._progress.setRange(0, 0)
        self._progress.setFixedHeight(5)
        self._progress.setTextVisible(False)
        self._progress.setVisible(False)
        layout.addWidget(self._progress)

        self._btn = QPushButton("Start Meeting")
        self._btn.setStyleSheet(_BTN_START)
        self._btn.clicked.connect(self._on_button)
        layout.addWidget(self._btn)

        self._couples_cb = QCheckBox("Couples session")
        layout.addWidget(self._couples_cb, alignment=Qt.AlignmentFlag.AlignHCenter)

        links = QHBoxLayout()
        self._sessions_btn = QPushButton("Sessions")
        self._sessions_btn.setCheckable(True)
        self._sessions_btn.toggled.connect(self._toggle_sessions)
        links.addWidget(self._sessions_btn)
        self._health_btn = QPushButton("Health Check")
        self._health_btn.clicked.connect(self._open_health)
        links.addWidget(self._health_btn)
        self._setup_btn = QPushButton("Setup && Test")
        self._setup_btn.clicked.connect(self._open_setup)
        links.addWidget(self._setup_btn)
        layout.addLayout(links)

        self._sessions_panel = QWidget()
        panel = QVBoxLayout(self._sessions_panel)
        panel.setContentsMargins(0, 6, 0, 0)
        self._list = QListWidget()
        self._list.setFixedHeight(190)
        self._list.currentItemChanged.connect(self._on_selection)
        self._list.itemDoubleClicked.connect(lambda _: self._view_selected())
        panel.addWidget(self._list)
        self._detail_lbl = QLabel()
        self._detail_lbl.setWordWrap(True)
        self._detail_lbl.setFixedWidth(388)
        panel.addWidget(self._detail_lbl)
        actions = QHBoxLayout()
        self._view_btn = QPushButton("View Notes")
        self._view_btn.clicked.connect(self._view_selected)
        actions.addWidget(self._view_btn)
        self._retry_btn = QPushButton("Retry")
        self._retry_btn.clicked.connect(self._retry_selected)
        actions.addWidget(self._retry_btn)
        self._folder_btn = QPushButton("Open Folder")
        self._folder_btn.clicked.connect(self._open_selected_folder)
        actions.addWidget(self._folder_btn)
        panel.addLayout(actions)
        self._sessions_panel.setVisible(False)
        layout.addWidget(self._sessions_panel)
        self._update_actions()

    def _set_status(self, text: str, color: Optional[str] = None) -> None:
        self._status_lbl.setText(text)
        self._status_lbl.setStyleSheet(f"font-size: 12px; color: {color};" if color else "font-size: 12px;")

    def _toggle_sessions(self, on: bool) -> None:
        self._sessions_panel.setVisible(on)
        self.adjustSize()

    # ── Config and setup ──────────────────────────────────────────────────────

    def _check_config(self) -> None:
        errors = config.validate()
        for error in errors:
            log.warning("Config: %s", error.split(":")[0])
        if self._recording or self._busy:
            return
        if errors:
            self._set_status("Setup required — click Setup & Test", "orange")
            self._btn.setEnabled(False)
        else:
            self._btn.setEnabled(True)
            if self._status_lbl.text().startswith("Setup required"):
                self._set_status("Ready")

    def _open_setup(self) -> None:
        dialog = SetupWindow(self)
        dialog.exec()
        dialog.deleteLater()
        reload_config()
        self._check_config()
        self._timer_lbl.setText(self._fmt(0))
        self._run_health()

    # ── Recording ─────────────────────────────────────────────────────────────

    def _on_button(self) -> None:
        if self._recording:
            self._stop()
        else:
            self._start()

    def _start(self) -> None:
        reload_config()
        if config.validate():
            self._check_config()
            return
        self._busy = True
        self._btn.setEnabled(False)
        self._btn.setText("Starting…")
        self._couples_cb.setEnabled(False)
        self._setup_btn.setEnabled(False)
        self._set_status("Starting recording…")
        self._recorder.start(self._couples_cb.isChecked())
        self._run_health()

    def _on_recording_started(self, session_id: str) -> None:
        log.info("Recording started: %s", session_id)
        self._recording = True
        self._busy = False
        self._started_at = time.monotonic()
        self._limit_warned = False
        self._btn.setEnabled(True)
        self._btn.setText("Stop Meeting")
        self._btn.setStyleSheet(_BTN_STOP)
        self._set_status("Recording…")
        self._tick_timer.start()
        self._refresh()

    def _on_recording_status(self, _message: dict) -> None:
        pass    # levels are available here for a future meter

    def _tick(self) -> None:
        elapsed = int(time.monotonic() - self._started_at)
        self._timer_lbl.setText(self._fmt(elapsed))
        if elapsed >= config.session_duration_minutes * 60 and not self._limit_warned:
            self._limit_warned = True
            QApplication.beep()
            self._set_status("Scheduled time reached. Still recording; stop when you're done.", "orange")

    def _stop(self) -> None:
        self._btn.setEnabled(False)
        self._btn.setText("Saving…")
        self._set_status("Saving recording…")
        self._recorder.stop()

    def _reset_controls(self) -> None:
        self._recording = False
        self._busy = False
        self._tick_timer.stop()
        self._timer_lbl.setText(self._fmt(0))
        self._btn.setText("Start Meeting")
        self._btn.setStyleSheet(_BTN_START)
        self._couples_cb.setEnabled(True)
        self._setup_btn.setEnabled(True)
        self._btn.setEnabled(True)
        self._check_config()

    def _on_recording_stopped(self, saved: bool) -> None:
        self._reset_controls()
        self._set_status("Saved. Notes will be written in the background." if saved else "Nothing was recorded.")
        if self._quit_after_stop:
            QApplication.quit()
            return
        self._refresh()
        self._run_health()

    def _on_recording_failed(self, message: str) -> None:
        log.error("Recording failed to start")
        self._reset_controls()
        self._set_status(message, "#e74c3c")
        self._quit_after_stop = False

    def _show_warning(self, message: str) -> None:
        self._set_status(message, "orange")
        QApplication.beep()

    # ── Sessions and worker ───────────────────────────────────────────────────

    def _refresh(self) -> None:
        try:
            sessions = store.list_sessions(config.sessions_dir, _SESSION_LIMIT)
        except OSError as exc:
            log.error("Could not read sessions: %s", exc)
            return
        self._ensure_worker(sessions)
        self._update_list(sessions)
        self._notice_finished(sessions)
        active = sum(1 for _, data in sessions if data.get("state") in store.PENDING)
        self._progress.setVisible(active > 0)
        if not self._recording and not self._busy and not self._status_lbl.text().startswith(("Setup required", "Could not")):
            self._show_idle_status(sessions, active)

    def _show_idle_status(self, sessions, active: int) -> None:
        current = next((d for _, d in sessions if d.get("state") in store.IN_PROGRESS), None)
        if current:
            waiting = active - 1
            text = _STATE_TEXT[current["state"]] + "…"
            self._set_status(f"{text} ({waiting} more waiting)" if waiting else text)
        elif active:
            self._set_status(f"{active} session(s) waiting to process")
        else:
            failed = sum(1 for _, d in sessions if d.get("state") == store.FAILED)
            if failed:
                self._set_status(f"{failed} session(s) need attention. Open Sessions to retry.", "#e74c3c")
            elif self._status_lbl.text().endswith("in the background.") or "attention" in self._status_lbl.text():
                self._set_status("Ready")

    def _ensure_worker(self, sessions) -> None:
        if self._worker is not None or self._recording or self._busy:
            return
        needs_work = any(
            data.get("state") in store.PENDING or data.get("state") == store.RECORDING
            for _, data in sessions
        )
        if not needs_work or time.monotonic() - self._worker_failed_at < _WORKER_RESTART_DELAY:
            return
        self._worker = make_process(self, "worker")
        self._worker.setProcessChannelMode(QProcess.ProcessChannelMode.ForwardedChannels)
        self._worker.finished.connect(self._on_worker_finished)
        self._worker.start()
        log.info("Worker started")

    def _on_worker_finished(self, code: int, status: QProcess.ExitStatus) -> None:
        if code != 0 or status != QProcess.ExitStatus.NormalExit:
            # Queue state is on disk, so restart after a pause and let it resume.
            log.error("Worker exited abnormally (code %s)", code)
            self._worker_failed_at = time.monotonic()
        self._worker.deleteLater()
        self._worker = None
        self._refresh()

    def _notice_finished(self, sessions) -> None:
        first_pass = not self._states
        for session_dir, data in sessions:
            previous = self._states.get(data["id"])
            state = data.get("state", "")
            self._states[data["id"]] = state
            if first_pass or previous is None or previous == state or state != store.DONE:
                continue
            if not self._recording:
                self._open_notes(session_dir, data)

    def _update_list(self, sessions) -> None:
        self._list.blockSignals(True)
        selected = self._selected_id
        self._list.clear()
        for session_dir, data in sessions:
            item = QListWidgetItem(session_label(data))
            item.setData(Qt.ItemDataRole.UserRole, str(session_dir))
            self._list.addItem(item)
            if data["id"] == selected:
                self._list.setCurrentItem(item)
        self._list.blockSignals(False)
        self._update_actions()

    def _selected(self) -> Optional[tuple[Path, dict]]:
        item = self._list.currentItem()
        if item is None:
            return None
        session_dir = Path(item.data(Qt.ItemDataRole.UserRole))
        return session_dir, store.load_session(session_dir)

    def _on_selection(self, current, _previous) -> None:
        self._selected_id = Path(current.data(Qt.ItemDataRole.UserRole)).name if current else None
        self._update_actions()

    def _update_actions(self) -> None:
        selected = self._selected()
        if selected is None:
            for button in (self._view_btn, self._retry_btn, self._folder_btn):
                button.setEnabled(False)
            self._detail_lbl.setVisible(False)
            return
        session_dir, data = selected
        state = data.get("state")
        self._view_btn.setEnabled((session_dir / store.NOTES_FILE).exists())
        self._retry_btn.setEnabled(state in (store.FAILED, store.UNPROCESSED))
        self._folder_btn.setEnabled(True)
        details = []
        if data.get("error"):
            details.append(str(data["error"]))
        details += data.get("warnings", [])
        self._detail_lbl.setText("\n".join(details))
        self._detail_lbl.setVisible(bool(details))
        self._detail_lbl.setStyleSheet("color: #e74c3c; font-size: 11px;" if data.get("error") else "color: orange; font-size: 11px;")
        self.adjustSize()

    def _retry_selected(self) -> None:
        selected = self._selected()
        if selected is None:
            return
        session_dir, data = selected
        couples = None
        if data.get("couples") is None:
            answer = QMessageBox.question(
                self, "Session type",
                "Was this a couples session?\n\nChoosing Yes uses the couples note template.",
            )
            couples = answer == QMessageBox.StandardButton.Yes
        store.mark_for_retry(session_dir, couples)
        self._worker_failed_at = 0.0
        self._refresh()

    def _open_selected_folder(self) -> None:
        selected = self._selected()
        if selected:
            subprocess.Popen(["open", str(selected[0])])

    def _view_selected(self) -> None:
        selected = self._selected()
        if selected and (selected[0] / store.NOTES_FILE).exists():
            self._open_notes(*selected)

    # ── Notes viewer ──────────────────────────────────────────────────────────

    def _open_notes(self, session_dir: Path, data: dict) -> None:
        notes_path = session_dir / store.NOTES_FILE
        try:
            notes = notes_path.read_text(encoding="utf-8")
        except OSError as exc:
            self._set_status(f"Could not open notes: {exc}", "#e74c3c")
            return

        dialog = QDialog(self)
        dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        dialog.setWindowTitle(f"Session Notes — {session_label(data).split(' — ')[0]}")
        dialog.resize(740, 680)
        layout = QVBoxLayout(dialog)

        notice = QLabel("AI draft. Review every section, especially risk assessment, before charting.")
        notice.setWordWrap(True)
        layout.addWidget(notice)
        for warning in data.get("warnings", []):
            label = QLabel(f"⚠ {warning}")
            label.setWordWrap(True)
            label.setStyleSheet("color: orange;")
            layout.addWidget(label)

        text = QTextEdit()
        text.setPlainText(notes)
        text.setFont(QFont("Helvetica", 13))
        layout.addWidget(text)

        buttons = QHBoxLayout()
        copy_btn = QPushButton("Copy to Clipboard")
        copy_btn.clicked.connect(lambda: QApplication.clipboard().setText(text.toPlainText()))
        buttons.addWidget(copy_btn)
        folder_btn = QPushButton("Open Session Folder")
        folder_btn.clicked.connect(lambda: subprocess.Popen(["open", str(session_dir)]))
        buttons.addWidget(folder_btn)
        buttons.addStretch()
        layout.addLayout(buttons)
        dialog.show()

    # ── Health check ──────────────────────────────────────────────────────────

    def _run_health(self) -> None:
        if self._health is not None:
            return
        self._health = make_process(self, "health")
        self._health.finished.connect(self._on_health_done)
        self._health.start()
        QTimer.singleShot(30_000, self._kill_stuck_health)

    def _kill_stuck_health(self) -> None:
        if self._health is not None:
            log.warning("Health check timed out")
            self._health.kill()

    def _on_health_done(self, *_args) -> None:
        process, self._health = self._health, None
        if process is None:
            return
        try:
            self._checks = json.loads(bytes(process.readAllStandardOutput()))
        except ValueError:
            self._checks = [{"name": "Health check", "level": "warn", "message": "The health check couldn't finish.", "fix": "report"}]
        process.deleteLater()
        self._update_banner()

    def _update_banner(self) -> None:
        worst = next((c for c in self._checks if c["level"] == "urgent"), None)
        worst = worst or next((c for c in self._checks if c["level"] == "warn" and c["name"] == "Disk space"), None)
        if worst is None:
            self._banner.setVisible(False)
        else:
            self._banner_lbl.setText(worst["message"])
            color = "#e74c3c" if worst["level"] == "urgent" else "orange"
            self._banner_lbl.setStyleSheet(f"color: {color}; font-size: 11px;")
            self._banner.setVisible(True)
        attention = sum(1 for c in self._checks if c["level"] != "ok")
        self._health_btn.setText(f"Health Check ({attention})" if attention else "Health Check")
        self.adjustSize()

    def _open_health(self) -> None:
        dialog = QDialog(self)
        dialog.setWindowTitle("Health Check")
        dialog.resize(520, 380)
        layout = QVBoxLayout(dialog)
        marks = {"ok": "✓", "warn": "⚠", "urgent": "✗"}
        colors = {"ok": "#27ae60", "warn": "orange", "urgent": "#e74c3c"}
        if not self._checks:
            layout.addWidget(QLabel("The health check is still running."))
        for check in self._checks:
            label = QLabel(f"{marks.get(check['level'], '')}  {check['name']}: {check['message']}")
            label.setWordWrap(True)
            label.setStyleSheet(f"color: {colors.get(check['level'], 'gray')};")
            layout.addWidget(label)
        layout.addStretch()

        buttons = QHBoxLayout()
        cleanup = QPushButton("Delete Audio From Finished Sessions")
        cleanup.clicked.connect(lambda: self._cleanup_audio(dialog))
        buttons.addWidget(cleanup)
        report = QPushButton("Copy Diagnostic Report")
        report.clicked.connect(lambda: self._copy_report(report))
        buttons.addWidget(report)
        layout.addLayout(buttons)
        dialog.exec()
        dialog.deleteLater()

    def _cleanup_audio(self, parent: QDialog) -> None:
        finished = [d for d, _ in store.list_sessions(config.sessions_dir) if (d / store.NOTES_FILE).exists() and store.audio_paths(d)]
        total = sum(store.audio_bytes(d) for d in finished)
        if not finished:
            QMessageBox.information(parent, "Nothing to delete", "No finished sessions have audio stored.")
            return
        answer = QMessageBox.question(
            parent, "Delete audio?",
            f"Permanently delete {total / 1024**3:.1f} GB of audio from {len(finished)} finished session(s)?\n\n"
            "Notes and transcripts are kept. Sessions without notes aren't touched. This can't be undone.",
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        freed = 0
        for session_dir in finished:
            try:
                freed += store.delete_audio(session_dir)
            except (OSError, ValueError) as exc:
                log.error("Could not delete audio in %s: %s", session_dir.name, exc)
        log.info("Cleanup freed %.1f GB", freed / 1024**3)
        QMessageBox.information(parent, "Done", f"Freed {freed / 1024**3:.1f} GB.")
        self._run_health()

    def _copy_report(self, button: QPushButton) -> None:
        button.setEnabled(False)
        button.setText("Building…")
        process = make_process(self, "health", "--report")

        def done(*_args) -> None:
            QApplication.clipboard().setText(bytes(process.readAllStandardOutput()).decode("utf-8", "replace"))
            button.setEnabled(True)
            button.setText("Copied")
            process.deleteLater()

        process.finished.connect(done)
        process.start()

    # ── Quit ──────────────────────────────────────────────────────────────────

    def closeEvent(self, event) -> None:
        if self._recording or self._busy:
            answer = QMessageBox.question(
                self, "Recording in progress",
                "A session is being recorded. Stop and save it, then quit?",
            )
            event.ignore()
            if answer == QMessageBox.StandardButton.Yes:
                self._quit_after_stop = True
                self._stop()
            return
        (data_dir() / "running.flag").unlink(missing_ok=True)
        event.accept()

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _fmt(self, elapsed: int) -> str:
        em, es = divmod(elapsed, 60)
        tm, ts = divmod(config.session_duration_minutes * 60, 60)
        return f"{em:02d}:{es:02d}  /  {tm:02d}:{ts:02d}"


def run() -> None:
    setup_logging("ui")
    app = QApplication(sys.argv)

    data_dir().mkdir(parents=True, exist_ok=True, mode=0o700)
    lock = QLockFile(str(data_dir() / "app.lock"))
    if not lock.tryLock(100):
        QMessageBox.information(None, "Therapy Notes", "Therapy Notes is already running.")
        sys.exit(0)

    flag = data_dir() / "running.flag"
    if flag.exists():
        log.warning("The previous run did not exit cleanly")
    flag.touch()

    log.info("App started")
    window = TherapyNotesApp()
    window.show()
    code = app.exec()
    flag.unlink(missing_ok=True)
    sys.exit(code)
