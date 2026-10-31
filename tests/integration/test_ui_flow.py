"""The real main window, offscreen, with stand-ins for the child processes."""
from __future__ import annotations

import json
import os
import sys
import textwrap
import time

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtCore import QEventLoop, QProcess, QTimer  # noqa: E402
from PyQt6.QtWidgets import QApplication, QMessageBox  # noqa: E402

from therapy_notes import session as store  # noqa: E402
from therapy_notes.ui import app as app_module  # noqa: E402

FAKE_RECORDER = textwrap.dedent("""
    import json, sys
    def send(**m): print(json.dumps(m), flush=True)
    mode = sys.argv[1]
    if mode == "fail":
        send(event="error", message="Could not start recording: no mic")
        sys.exit(1)
    send(event="started", session="fake")
    send(event="warning", code="silent", message="The microphone has been silent")
    if mode == "crash":
        sys.exit(9)
    for line in sys.stdin:
        if line.strip() == "stop":
            break
    send(event="stopped", session="fake", seconds=5.0)
""")


@pytest.fixture(scope="module")
def qapp():
    yield QApplication.instance() or QApplication([])


def spin(condition, timeout=8.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        QApplication.processEvents(QEventLoop.ProcessEventsFlag.AllEvents, 50)
        if condition():
            return True
        time.sleep(0.01)
    return False


@pytest.fixture
def window(qapp, tmp_path, monkeypatch, mock_config):
    cfg = mock_config(SESSIONS_DIR=str(tmp_path / "sessions"))
    (tmp_path / "sessions").mkdir()
    monkeypatch.setenv("THERAPY_NOTES_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setattr(app_module, "config", cfg)
    monkeypatch.setattr(app_module, "reload_config", lambda: cfg)
    state = {"mode": "normal", "workers": 0}

    def fake_process(parent, module, *args):
        process = QProcess(parent)
        process.setProgram(sys.executable)
        if module == "recorder_service":
            process.setArguments(["-c", FAKE_RECORDER, state["mode"]])
        elif module == "health":
            checks = json.dumps(state.get("checks", []))
            process.setArguments(["-c", f"print({checks!r})"])
        else:
            state["workers"] += 1
            process.setArguments(["-c", "pass"])
        return process

    monkeypatch.setattr(app_module, "make_process", fake_process)
    w = app_module.TherapyNotesApp()
    w.fake = state
    yield w
    w._poll_timer.stop()
    w._health_timer.stop()
    w._tick_timer.stop()
    w.deleteLater()


def test_start_stop_flow(window):
    window._on_button()
    assert window._btn.text() == "Starting…"
    assert spin(lambda: window._recording)
    assert window._btn.text() == "Stop Meeting"
    assert not window._setup_btn.isEnabled()
    window._on_button()
    assert spin(lambda: not window._recording)
    assert window._btn.text() == "Start Meeting" and window._btn.isEnabled()
    assert window._setup_btn.isEnabled()


def test_start_failure_is_shown_and_doesnt_crash(window):
    window.fake["mode"] = "fail"
    window._on_button()
    assert spin(lambda: "no mic" in window._status_lbl.text())
    assert window._btn.isEnabled() and window._btn.text() == "Start Meeting"
    assert not window._recording


def test_recorder_crash_resets_and_warns(window):
    window.fake["mode"] = "crash"
    window._on_button()
    assert spin(lambda: window._btn.text() == "Start Meeting" and not window._recording)
    assert window._btn.isEnabled()


def test_no_auto_stop_at_time_limit(window, mock_config):
    window._on_button()
    assert spin(lambda: window._recording)
    window._started_at = time.monotonic() - 100 * 3600
    window._tick()
    assert window._recording and "Still recording" in window._status_lbl.text()
    window._on_button()
    assert spin(lambda: not window._recording)


def test_quit_during_recording_asks_and_saves(window, monkeypatch):
    window._on_button()
    assert spin(lambda: window._recording)
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.No)

    class Event:
        ignored = False
        def ignore(self): self.ignored = True
        def accept(self): raise AssertionError("must not close while recording")

    event = Event()
    window.closeEvent(event)
    assert event.ignored and window._recording
    window._on_button()
    assert spin(lambda: not window._recording)


def test_worker_starts_for_queued_session_but_not_while_recording(window, tmp_path):
    root = tmp_path / "sessions"
    window._on_button()
    assert spin(lambda: window._recording)
    d = root / "2026-01-01_00-00-00"
    d.mkdir()
    store.update_session(d, state=store.QUEUED)
    window._refresh()
    assert window.fake["workers"] == 0
    window._on_button()
    assert spin(lambda: not window._recording)
    assert spin(lambda: window.fake["workers"] >= 1)


def test_session_list_and_failed_retry(window, tmp_path):
    d = tmp_path / "sessions" / "2026-01-01_00-00-00"
    d.mkdir()
    store.update_session(d, state=store.FAILED, error="The API key was rejected.", couples=False)
    window._refresh()
    assert "Failed" in window._list.item(0).text()
    window._list.setCurrentRow(0)
    assert window._retry_btn.isEnabled() and "API key" in window._detail_lbl.text()
    window._retry_selected()
    assert store.load_session(d)["state"] == store.QUEUED


def test_health_banner_shows_urgent_and_clears(window):
    assert spin(lambda: window._health is None)
    window.fake["checks"] = [{"name": "Disk space", "level": "urgent", "message": "Only 1.0 GB free.", "fix": "cleanup"}]
    window._run_health()
    assert spin(lambda: "1.0 GB" in window._banner_lbl.text() and not window._banner.isHidden())
    window.fake["checks"] = []
    assert spin(lambda: window._health is None)
    window._run_health()
    assert spin(lambda: window._banner.isHidden())


def test_config_fix_reenables_start_without_restart(window, monkeypatch, mock_config):
    bad = mock_config(ANTHROPIC_API_KEY="")
    monkeypatch.setattr(app_module, "config", bad)
    window._check_config()
    assert not window._btn.isEnabled()
    bad.anthropic_api_key = "sk-real"
    window._check_config()
    assert window._btn.isEnabled() and window._status_lbl.text() == "Ready"
