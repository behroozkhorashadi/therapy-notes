"""Worker: crash recovery, retry limits, and pausing during a recording."""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import textwrap
import threading
import time
import wave
from pathlib import Path

import pytest

from therapy_notes import session as store, worker
from therapy_notes.audio.recorder import SAMPLE_RATE

SRC = Path(worker.__file__).resolve().parents[1]


def write_wav(path, seconds=2):
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(SAMPLE_RATE)
        handle.writeframes(b"\0\1" * (SAMPLE_RATE * seconds))


def test_recovers_session_from_killed_recorder(tmp_path):
    """Kill a real recording process with SIGKILL, then recover its audio."""
    root = tmp_path / "sessions"
    script = textwrap.dedent(f"""
        import sys, time
        import numpy as np
        from pathlib import Path
        from therapy_notes import session as store
        from therapy_notes.audio.recorder import StreamingRecorder, SAMPLE_RATE
        root = Path({str(root)!r})
        d = store.create_session(root, couples=False)
        store.write_marker(root, d.name)
        class S:
            def start(self): pass
            def stop(self): pass
            def close(self): pass
        cb = {{}}
        def factory(device, callback):
            cb["f"] = callback
            return S()
        r = StreamingRecorder(d, None, stream_factory=factory, interval=0.02)
        r.start()
        cb["f"](np.ones((SAMPLE_RATE * 5, 1), dtype=np.int16), SAMPLE_RATE * 5, None, None)
        while r.tracks[0].frames == 0:
            time.sleep(0.01)
        print(d, flush=True)
        time.sleep(60)
    """)
    env = {**os.environ, "PYTHONPATH": str(SRC)}
    child = subprocess.Popen([sys.executable, "-c", script], stdout=subprocess.PIPE, text=True, env=env)
    session_dir = Path(child.stdout.readline().strip())
    assert store.load_session(session_dir)["state"] == store.RECORDING
    assert store.active_recording(root) == session_dir.name

    worker.recover_interrupted_recordings(root)     # still alive: must leave it alone
    assert store.load_session(session_dir)["state"] == store.RECORDING

    os.kill(child.pid, signal.SIGKILL)
    child.wait()
    child.stdout.close()
    worker.recover_interrupted_recordings(root)
    data = store.load_session(session_dir)
    assert data["state"] == store.QUEUED
    assert data["seconds"] == pytest.approx(5.0)
    assert any("ended unexpectedly" in w for w in data["warnings"])
    with wave.open(str(session_dir / store.THERAPIST_AUDIO)) as handle:
        assert handle.getnframes() == SAMPLE_RATE * 5


def test_recording_with_no_audio_is_marked_failed(tmp_path):
    d = store.create_session(tmp_path, couples=False)
    worker.recover_interrupted_recordings(tmp_path)
    assert store.load_session(d)["state"] == store.FAILED


def test_oldest_job_first_and_interrupted_jobs_resume(tmp_path):
    for name, state in (("2026-01-03_00-00-00", store.QUEUED), ("2026-01-02_00-00-00", store.TRANSCRIBING),
                        ("2026-01-01_00-00-00", store.DONE), ("2025-12-01_00-00-00", store.UNPROCESSED)):
        (tmp_path / name).mkdir()
        store.update_session(tmp_path / name, state=state)
    assert worker.next_job(tmp_path).name == "2026-01-02_00-00-00"


def test_failed_jobs_are_not_retried_automatically(tmp_path):
    d = tmp_path / "2026-01-01_00-00-00"
    d.mkdir()
    store.update_session(d, state=store.FAILED)
    assert worker.next_job(tmp_path) is None


def make_run(root, monkeypatch, process):
    monkeypatch.setattr(worker, "process_session", process)
    return worker._Run(root)


def test_error_marks_failed_and_queue_continues(tmp_path, monkeypatch):
    d = store.create_session(tmp_path, False)
    store.update_session(d, state=store.QUEUED)

    def boom(*_):
        store.update_session(d, state=store.GENERATING)
        raise type("AuthenticationError", (Exception,), {})("401")

    make_run(tmp_path, monkeypatch, boom).run_job(d)
    data = store.load_session(d)
    assert data["state"] == store.FAILED
    assert data["failed_stage"] == store.GENERATING
    assert "API key" in data["error"]


def test_repeated_interruptions_stop_a_crash_loop(tmp_path, monkeypatch):
    d = store.create_session(tmp_path, False)
    store.update_session(d, state=store.TRANSCRIBING, attempts=worker.MAX_ATTEMPTS)
    called = []
    make_run(tmp_path, monkeypatch, lambda *_: called.append(1)).run_job(d)
    assert not called
    assert store.load_session(d)["state"] == store.FAILED


def test_app_closing_requeues_without_using_an_attempt(tmp_path, monkeypatch):
    d = store.create_session(tmp_path, False)
    store.update_session(d, state=store.TRANSCRIBING)

    def closed(*_):
        raise worker._AppClosed

    with pytest.raises(worker._AppClosed):
        make_run(tmp_path, monkeypatch, closed).run_job(d)
    data = store.load_session(d)
    assert data["state"] == store.QUEUED and data["attempts"] == 0


def test_pauses_while_recording_then_resumes(tmp_path, monkeypatch):
    monkeypatch.setattr(worker, "_POLL_SECONDS", 0.01)
    run = worker._Run(tmp_path)
    store.write_marker(tmp_path, "s")
    finished = threading.Event()
    thread = threading.Thread(target=lambda: (run.wait(), finished.set()), daemon=True)
    thread.start()
    time.sleep(0.1)
    assert not finished.is_set()
    store.clear_marker(tmp_path)
    assert finished.wait(2)


def test_wait_detects_app_closed(tmp_path, monkeypatch):
    run = worker._Run(tmp_path)
    monkeypatch.setattr(worker.os, "getppid", lambda: 1)
    with pytest.raises(worker._AppClosed):
        run.wait()
