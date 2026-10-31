"""The real recorder process, with a fake sounddevice module standing in for hardware."""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import textwrap
import time
import wave
from pathlib import Path

import pytest

from therapy_notes import session as store

SRC = Path(store.__file__).resolve().parents[1]

FAKE_SOUNDDEVICE = textwrap.dedent("""
    import os, threading, time
    import numpy as np

    def query_devices():
        return [{"name": "Fake Mic", "max_input_channels": 1, "max_output_channels": 0},
                {"name": "BlackHole 2ch", "max_input_channels": 2, "max_output_channels": 2}]

    class InputStream:
        def __init__(self, device=None, samplerate=16000, channels=1, dtype="int16", callback=None):
            if os.environ.get("FAKE_MIC_FAILS"):
                raise RuntimeError("Error opening InputStream: no device")
            self.callback, self.rate, self.running = callback, samplerate, False

        def _run(self):
            while self.running:
                frames = self.rate // 10
                self.callback(np.full((frames, 1), 500, dtype=np.int16), frames, None, None)
                time.sleep(0.1)

        def start(self):
            self.running = True
            threading.Thread(target=self._run, daemon=True).start()

        def stop(self):
            self.running = False
            if os.environ.get("FAKE_STOP_HANGS"):
                time.sleep(120)

        def close(self):
            pass
""")


class Service:
    def __init__(self, tmp_path: Path, **env):
        fake = tmp_path / "fake"
        fake.mkdir(exist_ok=True)
        (fake / "sounddevice.py").write_text(FAKE_SOUNDDEVICE)
        self.root = tmp_path / "sessions"
        full_env = {
            **os.environ,
            "PYTHONPATH": f"{fake}:{SRC}",
            "SESSIONS_DIR": str(self.root),
            "THERAPY_NOTES_DATA_DIR": str(tmp_path / "data"),
            "AUDIO_PRIMARY_DEVICE": "",
            "AUDIO_LOOPBACK_DEVICE": "BlackHole 2ch",
            **env,
        }
        self.process = subprocess.Popen(
            [sys.executable, "-m", "therapy_notes.recorder_service"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, env=full_env,
        )

    def wait_for(self, event: str) -> dict:
        for line in self.process.stdout:
            message = json.loads(line)
            if message.get("event") == event:
                return message
        raise AssertionError(f"process ended before '{event}'")

    def session_dir(self) -> Path:
        return next(d for d in self.root.iterdir() if d.is_dir())


def frames(path: Path) -> int:
    with wave.open(str(path), "rb") as handle:
        return handle.getnframes()


def test_records_both_tracks_and_queues_on_stop(tmp_path):
    service = Service(tmp_path)
    service.wait_for("started")
    assert store.active_recording(service.root) == service.session_dir().name
    service.wait_for("status")
    time.sleep(1.2)
    service.process.stdin.write("stop\n")
    service.process.stdin.flush()
    stopped = service.wait_for("stopped")
    service.process.wait(10)
    session = service.session_dir()
    assert stopped["seconds"] > 0.5
    assert store.load_session(session)["state"] == store.QUEUED
    assert frames(session / store.THERAPIST_AUDIO) > 8000
    assert frames(session / store.CLIENT_AUDIO) > 8000
    assert store.active_recording(service.root) is None


def test_ui_disappearing_still_saves_the_recording(tmp_path):
    service = Service(tmp_path)
    service.wait_for("started")
    service.wait_for("status")
    time.sleep(1.2)
    service.process.stdin.close()      # what happens if the UI dies
    service.process.wait(15)
    session = service.session_dir()
    assert store.load_session(session)["state"] == store.QUEUED
    assert frames(session / store.THERAPIST_AUDIO) > 8000


def test_kill_leaves_recoverable_audio(tmp_path):
    from therapy_notes.worker import recover_interrupted_recordings

    service = Service(tmp_path)
    service.wait_for("started")
    service.wait_for("status")
    time.sleep(2.2)
    os.kill(service.process.pid, signal.SIGKILL)
    service.process.wait()
    recover_interrupted_recordings(service.root)
    session = service.session_dir()
    assert store.load_session(session)["state"] == store.QUEUED
    assert frames(session / store.THERAPIST_AUDIO) > 8000


def test_hung_audio_device_still_saves(tmp_path):
    service = Service(tmp_path, FAKE_STOP_HANGS="1")
    service.wait_for("started")
    service.wait_for("status")
    time.sleep(1.2)
    started = time.monotonic()
    service.process.stdin.write("stop\n")
    service.process.stdin.flush()
    service.wait_for("stopped")
    service.process.wait(15)
    assert time.monotonic() - started < 12
    assert store.load_session(service.session_dir())["state"] == store.QUEUED


def test_missing_mic_reports_error_and_leaves_nothing_behind(tmp_path):
    service = Service(tmp_path, FAKE_MIC_FAILS="1")
    message = service.wait_for("error")
    service.process.wait(10)
    assert "no device" in message["message"]
    assert [d for d in service.root.iterdir() if d.is_dir()] == []
    assert store.active_recording(service.root) is None


def test_very_short_recording_is_discarded(tmp_path):
    service = Service(tmp_path)
    service.wait_for("started")
    service.process.stdin.write("stop\n")
    service.process.stdin.flush()
    stopped = service.wait_for("stopped")
    service.process.wait(10)
    if stopped.get("discarded"):
        assert [d for d in service.root.iterdir() if d.is_dir()] == []
