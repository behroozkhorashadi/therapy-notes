"""Crash-safe audio capture: writes each track to its WAV file as it records.

Audio callbacks only copy data into a queue. A writer thread saves it about once
a second, keeping the WAV header valid, so a crash loses at most the last moment.

macOS setup for capturing the client's side of a call is covered by the Setup
window (BlackHole 2ch plus a Multi-Output Device).
"""
from __future__ import annotations

import logging
import os
import queue
import shutil
import struct
import threading
import time
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from ..session import CLIENT_AUDIO, THERAPIST_AUDIO

log = logging.getLogger(__name__)

SAMPLE_RATE = 16_000
CHANNELS = 1
DTYPE = "int16"
SAMPLE_WIDTH = 2

STALL_SECONDS = 5.0
MIC_SILENCE_SECONDS = 60.0
CLIENT_SILENCE_SECONDS = 120.0
LOW_DISK_BYTES = 1024**3
STOP_DISK_BYTES = 200 * 1024**2
_SILENCE_PEAK = 8       # int16 peak at or below this counts as silence
_STOP_TIMEOUT = 5.0     # macOS audio teardown can hang when a device has vanished

Emit = Callable[[str, str, str], None]      # kind, code, message
StreamFactory = Callable[..., object]


def _default_stream_factory(device, callback):
    import sounddevice as sd
    return sd.InputStream(
        device=device, samplerate=SAMPLE_RATE, channels=CHANNELS, dtype=DTYPE, callback=callback,
    )


@dataclass
class _Track:
    label: str
    device: object
    path: Path
    required: bool
    silence_limit: float
    silence_message: str
    frames: int = 0
    overflows: int = 0
    dropped: int = 0
    peak: int = 0
    silent_for: float = 0.0
    silence_warned: bool = False
    stall_warned: bool = False
    last_callback: float = 0.0
    queue: "queue.Queue[np.ndarray]" = field(default_factory=lambda: queue.Queue(maxsize=2000))
    stream: object = None
    wav: Optional[wave.Wave_write] = None

    def callback(self, indata, frames, time_info, status) -> None:
        self.last_callback = time.monotonic()
        if status:
            self.overflows += 1
        try:
            self.queue.put_nowait(indata.copy())
        except queue.Full:
            self.dropped += frames


class StreamingRecorder:
    def __init__(
        self,
        session_dir: Path,
        primary_device=None,
        loopback_device=None,
        emit: Optional[Emit] = None,
        stream_factory: StreamFactory = _default_stream_factory,
        interval: float = 1.0,
    ) -> None:
        self.session_dir = session_dir
        self._emit = emit or (lambda kind, code, message: None)
        self._factory = stream_factory
        self._interval = interval
        self._stop_writer = threading.Event()
        self._writer: Optional[threading.Thread] = None
        self.fatal: Optional[str] = None
        self.tracks = [_Track(
            "Microphone", primary_device, session_dir / THERAPIST_AUDIO, True, MIC_SILENCE_SECONDS,
            "The microphone has been silent for a minute. Check that it isn't muted.",
        )]
        if loopback_device is not None:
            self.tracks.append(_Track(
                "Client audio", loopback_device, session_dir / CLIENT_AUDIO, False, CLIENT_SILENCE_SECONDS,
                "No client audio for two minutes. Check that Zoom's speaker is set to Therapy Output.",
            ))

    def start(self) -> None:
        """Open the streams. Raises if the microphone can't start."""
        for track in list(self.tracks):
            try:
                self._open(track)
            except Exception as exc:  # noqa: BLE001 - audio errors have no shared base class
                self._close_wav(track)
                track.path.unlink(missing_ok=True)
                if track.required:
                    self._teardown()
                    raise
                log.warning("Could not open client audio: %s", exc)
                self._emit("warning", "client_unavailable",
                           "Client audio could not start, so only the microphone is recording. Check Setup.")
                self.tracks.remove(track)
        self._writer = threading.Thread(target=self._write_loop, name="writer", daemon=True)
        self._writer.start()

    def _open(self, track: _Track) -> None:
        track.wav = wave.open(str(track.path), "wb")
        track.wav.setnchannels(CHANNELS)
        track.wav.setsampwidth(SAMPLE_WIDTH)
        track.wav.setframerate(SAMPLE_RATE)
        track.last_callback = time.monotonic()
        track.stream = self._factory(track.device, track.callback)
        track.stream.start()

    def _write_loop(self) -> None:
        last_sync = time.monotonic()
        last_disk = 0.0
        while not self._stop_writer.wait(self._interval):
            try:
                self._drain()
                now = time.monotonic()
                if now - last_sync >= 10:
                    self._sync()
                    last_sync = now
                if now - last_disk >= 30:
                    self._check_disk()
                    last_disk = now
                self._check_tracks(now)
                self._emit("status", "", "")
            except OSError as exc:
                self.fatal = f"Could not save audio: {exc}"
                log.error(self.fatal)
                self._emit("fatal", "write_failed", self.fatal)
                return

    def _drain(self) -> float:
        """Write queued audio. Returns the seconds written for the first track."""
        written = 0.0
        for index, track in enumerate(self.tracks):
            chunks = []
            while True:
                try:
                    chunks.append(track.queue.get_nowait())
                except queue.Empty:
                    break
            track.peak = 0
            if not chunks:
                continue
            data = np.concatenate(chunks)
            track.wav.writeframes(data.tobytes())
            track.frames += len(data)
            track.peak = int(np.abs(data.astype(np.int32)).max())
            seconds = len(data) / SAMPLE_RATE
            track.silent_for = track.silent_for + seconds if track.peak <= _SILENCE_PEAK else 0.0
            if index == 0:
                written = seconds
        return written

    def _sync(self) -> None:
        for track in self.tracks:
            handle = getattr(track.wav, "_file", None)
            if handle is not None:
                handle.flush()
                os.fsync(handle.fileno())

    def _check_disk(self) -> None:
        free = shutil.disk_usage(self.session_dir).free
        if free < STOP_DISK_BYTES:
            self.fatal = "The disk is nearly full, so recording stopped. The audio so far is saved."
            self._emit("fatal", "disk_full", self.fatal)
            self._stop_writer.set()
        elif free < LOW_DISK_BYTES:
            self._emit("warning", "disk_low", "Less than 1 GB of disk space left. Free up space soon.")

    def _check_tracks(self, now: float) -> None:
        for track in self.tracks:
            if now - track.last_callback > STALL_SECONDS:
                if not track.stall_warned:
                    track.stall_warned = True
                    log.warning("%s stopped delivering audio", track.label)
                    self._emit("warning", "stalled",
                               f"{track.label} stopped sending audio. The device may have been disconnected.")
            else:
                track.stall_warned = False
            if track.silent_for >= track.silence_limit:
                if not track.silence_warned:
                    track.silence_warned = True
                    self._emit("warning", "silent", track.silence_message)
            else:
                track.silence_warned = False

    def _teardown(self) -> None:
        """Stop the streams, without letting a hung audio device block saving."""
        def close_streams() -> None:
            for track in self.tracks:
                if track.stream is None:
                    continue
                try:
                    track.stream.stop()
                    track.stream.close()
                except Exception as exc:  # noqa: BLE001
                    log.warning("Error closing %s stream: %s", track.label, exc)

        closer = threading.Thread(target=close_streams, daemon=True)
        closer.start()
        closer.join(_STOP_TIMEOUT)
        if closer.is_alive():
            log.warning("Audio device did not close in time; saving anyway")

    def _close_wav(self, track: _Track) -> None:
        if track.wav is not None:
            try:
                track.wav.close()
            except Exception as exc:  # noqa: BLE001
                log.warning("Error closing %s file: %s", track.label, exc)
            track.wav = None

    def stop(self) -> dict:
        """Stop and save everything. Returns a summary of what was captured."""
        self._teardown()
        self._stop_writer.set()
        if self._writer is not None and self._writer is not threading.current_thread():
            self._writer.join(_STOP_TIMEOUT)
        try:
            self._drain()
        except OSError as exc:
            self.fatal = f"Could not save audio: {exc}"
        summary: dict = {"warnings": []}
        for track in self.tracks:
            self._close_wav(track)
            if track.overflows or track.dropped:
                log.warning("%s: %d overflows, %d dropped frames", track.label, track.overflows, track.dropped)
                summary["warnings"].append(
                    f"{track.label} had brief audio dropouts during recording, so the transcript may have gaps."
                )
        summary["seconds"] = self.tracks[0].frames / SAMPLE_RATE
        return summary

    def levels(self) -> dict:
        return {
            "peaks": [track.peak / 32768 for track in self.tracks],
            "seconds": self.tracks[0].frames / SAMPLE_RATE,
        }


def repair_wav(path: Path) -> float:
    """Make a WAV's header match its data after a crash. Returns its seconds."""
    size = path.stat().st_size
    if size < 44:
        raise ValueError(f"{path.name} is too small to be audio")
    data_bytes = size - 44
    with open(path, "r+b") as handle:
        header = handle.read(44)
        if header[:4] != b"RIFF" or header[36:40] != b"data":
            raise ValueError(f"{path.name} is not a recoverable WAV file")
        handle.seek(4)
        handle.write(struct.pack("<I", size - 8))
        handle.seek(40)
        handle.write(struct.pack("<I", data_bytes))
    return data_bytes / (SAMPLE_RATE * SAMPLE_WIDTH)
