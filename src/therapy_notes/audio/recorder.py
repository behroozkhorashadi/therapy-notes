from __future__ import annotations

"""
Audio recorder that captures one or two input streams and saves them as
separate WAV files (therapist mic + client loopback).

macOS setup — capturing BOTH sides of a Zoom / Google Meet call
================================================================
By default, macOS only lets apps record the microphone. To also capture the
client's voice coming through your speakers, you need a free virtual audio
driver called BlackHole.

One-time setup (takes ~5 minutes):

1. Download and install BlackHole 2ch (free):
   https://existential.audio/blackhole/

2. Open "Audio MIDI Setup" (search in Spotlight).
   Click the + button at the bottom-left → "Create Multi-Output Device".
   In the right panel, check BOTH:
     • Your normal speakers / headphones
     • BlackHole 2ch
   Rename it something like "Therapy Output".

3. System Settings → Sound → Output → select "Therapy Output".
   (You will still hear audio normally through your real speakers.)

4. In Zoom (and Google Meet), set Audio Output to "Therapy Output".

5. In your .env file, set:
     AUDIO_LOOPBACK_DEVICE=BlackHole 2ch

Run `uv run python tools/list_audio_devices.py` to confirm the device name.

That's it. The recorder saves your microphone as audio_therapist.wav and
the client's Zoom audio as audio_client.wav. The transcriber then produces
a speaker-labelled transcript: Therapist: … / Client: …
"""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import scipy.io.wavfile as wavfile
import sounddevice as sd


@dataclass
class RecordingResult:
    """Paths produced by a single recording session."""
    therapist_path: Path        # primary mic WAV
    client_path: Optional[Path] # loopback WAV — None when no loopback device
    session_dir: Path

SAMPLE_RATE = 16_000   # 16 kHz — matches Whisper's native rate, sufficient for speech
CHANNELS = 1
DTYPE = "int16"


class AudioRecorder:
    def __init__(
        self,
        output_dir: Path,
        primary_device: Optional[str | int] = None,
        loopback_device: Optional[str | int] = None,
        on_level: Optional[Callable[[float], None]] = None,
    ) -> None:
        self.output_dir = output_dir
        self.primary_device = primary_device    # None = system default mic
        self.loopback_device = loopback_device  # e.g. "BlackHole 2ch"
        self.on_level = on_level                # callback(rms: float 0–1)

        self._recording = False
        self._primary_chunks: list[np.ndarray] = []
        self._loopback_chunks: list[np.ndarray] = []
        self._primary_stream: Optional[sd.InputStream] = None
        self._loopback_stream: Optional[sd.InputStream] = None
        self._session_dir: Optional[Path] = None

    # ── Public interface ──────────────────────────────────────────────────────

    def start(self) -> Path:
        """Begin recording. Returns the session directory path."""
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        session_dir = self.output_dir / timestamp
        session_dir.mkdir(parents=True, exist_ok=True)
        self._session_dir = session_dir

        self._recording = True
        self._primary_chunks = []
        self._loopback_chunks = []

        self._primary_stream = sd.InputStream(
            device=self.primary_device,
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype=DTYPE,
            callback=self._primary_cb,
        )
        self._primary_stream.start()

        if self.loopback_device is not None:
            self._loopback_stream = sd.InputStream(
                device=self.loopback_device,
                samplerate=SAMPLE_RATE,
                channels=CHANNELS,
                dtype=DTYPE,
                callback=self._loopback_cb,
            )
            self._loopback_stream.start()

        return self._session_dir

    def stop(self) -> RecordingResult:
        """Stop recording, write WAV files, and return a RecordingResult."""
        self._recording = False

        for stream in (self._primary_stream, self._loopback_stream):
            if stream:
                stream.stop()
                stream.close()
        self._primary_stream = None
        self._loopback_stream = None

        session_dir = self._session_dir

        primary = (
            np.concatenate(self._primary_chunks, axis=0)
            if self._primary_chunks
            else np.zeros((SAMPLE_RATE, 1), dtype=DTYPE)
        )
        therapist_path = session_dir / "audio_therapist.wav"
        wavfile.write(str(therapist_path), SAMPLE_RATE, primary.flatten())

        client_path: Optional[Path] = None
        if self._loopback_chunks:
            loopback = np.concatenate(self._loopback_chunks, axis=0)
            client_path = session_dir / "audio_client.wav"
            wavfile.write(str(client_path), SAMPLE_RATE, loopback.flatten())

        return RecordingResult(
            therapist_path=therapist_path,
            client_path=client_path,
            session_dir=session_dir,
        )

    @property
    def session_dir(self) -> Optional[Path]:
        return self._session_dir

    # ── Stream callbacks (called from audio thread) ───────────────────────────

    def _primary_cb(self, indata: np.ndarray, frames: int, time, status) -> None:
        if not self._recording:
            return
        chunk = indata.copy()
        self._primary_chunks.append(chunk)
        if self.on_level:
            rms = float(np.sqrt(np.mean(chunk.astype(np.float32) ** 2))) / 32768.0
            self.on_level(min(rms * 12, 1.0))

    def _loopback_cb(self, indata: np.ndarray, frames: int, time, status) -> None:
        if self._recording:
            self._loopback_chunks.append(indata.copy())

    # ── Utility ───────────────────────────────────────────────────────────────

    @staticmethod
    def list_devices() -> None:
        """Print all available audio devices. Run tools/list_audio_devices.py."""
        print(sd.query_devices())
