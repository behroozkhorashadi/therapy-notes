"""Streaming recorder, driven by fake audio streams (no hardware)."""
from __future__ import annotations

import shutil
import time
import wave
from collections import namedtuple

import numpy as np
import pytest

from therapy_notes.audio import recorder as rec
from therapy_notes.audio.recorder import SAMPLE_RATE, StreamingRecorder, repair_wav


class FakeStream:
    def __init__(self, callback, hang=False):
        self.callback = callback
        self.hang = hang
        self.closed = False

    def start(self):
        pass

    def stop(self):
        if self.hang:
            time.sleep(60)

    def close(self):
        self.closed = True

    def feed(self, seconds=1.0, value=1000, status=None):
        frames = int(SAMPLE_RATE * seconds)
        self.callback(np.full((frames, 1), value, dtype=np.int16), frames, None, status)


def make(tmp_path, loopback=None, fail_on=(), hang=False, emit=None):
    streams = {}

    def factory(device, callback):
        if device in fail_on:
            raise RuntimeError(f"no device {device}")
        streams[device] = FakeStream(callback, hang)
        return streams[device]

    recorder = StreamingRecorder(tmp_path, "mic", loopback, emit=emit, stream_factory=factory, interval=0.02)
    return recorder, streams


def frames_in(path):
    with wave.open(str(path), "rb") as handle:
        assert handle.getframerate() == SAMPLE_RATE
        return handle.getnframes()


def wait_for(condition, timeout=3.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if condition():
            return True
        time.sleep(0.01)
    return False


def test_all_audio_saved_on_stop(tmp_path):
    r, streams = make(tmp_path, "loop")
    r.start()
    streams["mic"].feed(2.0)
    streams["loop"].feed(1.0)
    summary = r.stop()
    assert frames_in(r.tracks[0].path) == 2 * SAMPLE_RATE
    assert frames_in(r.tracks[1].path) == SAMPLE_RATE
    assert summary["seconds"] == pytest.approx(2.0)
    assert summary["warnings"] == []


def test_file_is_readable_while_recording_so_a_crash_keeps_audio(tmp_path):
    """Copying the file mid-recording stands in for the process dying at that moment."""
    r, streams = make(tmp_path)
    r.start()
    streams["mic"].feed(3.0)
    assert wait_for(lambda: r.tracks[0].frames == 3 * SAMPLE_RATE)
    time.sleep(0.05)
    crashed = tmp_path / "copy.wav"
    shutil.copy(r.tracks[0].path, crashed)
    assert frames_in(crashed) == 3 * SAMPLE_RATE
    r.stop()


def test_memory_is_not_held_after_writing(tmp_path):
    r, streams = make(tmp_path)
    r.start()
    streams["mic"].feed(1.0)
    assert wait_for(lambda: r.tracks[0].queue.empty() and r.tracks[0].frames > 0)
    r.stop()


def test_client_failure_keeps_mic_recording(tmp_path):
    events = []
    r, streams = make(tmp_path, "loop", fail_on=("loop",), emit=lambda *e: events.append(e))
    r.start()
    assert [t.label for t in r.tracks] == ["Microphone"]
    assert any(e[1] == "client_unavailable" for e in events)
    streams["mic"].feed(1.0)
    r.stop()
    assert not (tmp_path / "audio_client.wav").exists()


def test_mic_failure_raises_and_leaves_nothing_running(tmp_path):
    r, _ = make(tmp_path, fail_on=("mic",))
    with pytest.raises(RuntimeError):
        r.start()


def test_hung_device_does_not_block_saving(tmp_path, monkeypatch):
    monkeypatch.setattr(rec, "_STOP_TIMEOUT", 0.1)
    r, streams = make(tmp_path, hang=True)
    r.start()
    streams["mic"].feed(1.0)
    started = time.monotonic()
    r.stop()
    assert time.monotonic() - started < 2
    assert frames_in(r.tracks[0].path) == SAMPLE_RATE


def test_overflow_is_reported(tmp_path):
    r, streams = make(tmp_path)
    r.start()
    streams["mic"].feed(1.0, status="input overflow")
    summary = r.stop()
    assert any("dropouts" in w for w in summary["warnings"])


def test_stall_warning(tmp_path, monkeypatch):
    monkeypatch.setattr(rec, "STALL_SECONDS", 0.05)
    events = []
    r, _ = make(tmp_path, emit=lambda *e: events.append(e))
    r.start()
    assert wait_for(lambda: any(e[1] == "stalled" for e in events))
    r.stop()


def test_silent_client_warning(tmp_path, monkeypatch):
    monkeypatch.setattr(rec, "CLIENT_SILENCE_SECONDS", 1.0)
    monkeypatch.setattr(rec, "STALL_SECONDS", 100)
    events = []
    r = None

    def factory(device, callback):
        return streams.setdefault(device, FakeStream(callback))

    streams = {}
    r = StreamingRecorder(tmp_path, "mic", "loop", emit=lambda *e: events.append(e), stream_factory=factory, interval=0.02)
    r.tracks[1].silence_limit = 1.0
    r.start()
    streams["mic"].feed(2.0, value=1000)
    streams["loop"].feed(2.0, value=0)
    assert wait_for(lambda: any(e[1] == "silent" and "Zoom" in e[2] for e in events))
    r.stop()


def test_nearly_full_disk_stops_cleanly(tmp_path, monkeypatch):
    Usage = namedtuple("Usage", "total used free")
    monkeypatch.setattr(rec.shutil, "disk_usage", lambda _: Usage(1, 1, 1000))
    events = []
    r, _ = make(tmp_path, emit=lambda *e: events.append(e))
    r.start()
    assert wait_for(lambda: any(e[0] == "fatal" and e[1] == "disk_full" for e in events))
    r.stop()


def test_write_error_is_fatal_not_silent(tmp_path):
    events = []
    r, streams = make(tmp_path, emit=lambda *e: events.append(e))
    r.start()

    class Broken:
        def writeframes(self, _):
            raise OSError(28, "No space left on device")

        def close(self):
            pass

    r.tracks[0].wav = Broken()
    streams["mic"].feed(1.0)
    assert wait_for(lambda: any(e[0] == "fatal" for e in events))
    r.stop()
    assert "No space left" in r.fatal


def test_repair_wav_after_header_never_updated(tmp_path):
    path = tmp_path / "a.wav"
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(SAMPLE_RATE)
        handle.writeframes(b"\0" * 2000)
    with open(path, "ab") as handle:
        handle.write(b"\1\0" * SAMPLE_RATE)    # data the header doesn't know about
    assert repair_wav(path) == pytest.approx(1 + 1000 / SAMPLE_RATE)
    assert frames_in(path) == SAMPLE_RATE + 1000


def test_repair_rejects_non_wav(tmp_path):
    path = tmp_path / "a.wav"
    path.write_bytes(b"x" * 100)
    with pytest.raises(ValueError):
        repair_wav(path)
