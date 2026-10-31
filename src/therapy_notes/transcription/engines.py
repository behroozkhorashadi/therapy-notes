"""Speech-to-text engines, behind one small interface.

An engine turns a WAV file into [(start_seconds, text), ...] and calls `tick`
often, which is how the worker pauses transcription during a recording.

Add a new engine here and register it in `create_engine`.
"""
from __future__ import annotations

import logging
import os
import wave
from pathlib import Path
from typing import Callable, Optional, Protocol

import numpy as np

from ..models import PARAKEET_MODEL, recommend_models

log = logging.getLogger(__name__)

Segments = list[tuple[float, str]]
Tick = Optional[Callable[[], None]]

# Whisper treats this as earlier transcript. Keep it short and free of diagnoses,
# drug names and risk terms: Whisper keeps only the last ~223 tokens, and prompt
# words can leak into the transcript during silence or unclear audio.
_WHISPER_PROMPT = (
    "Telehealth therapy session between a therapist and a client. They may discuss "
    "CBT, DBT, EMDR, mindfulness, coping skills, and the treatment plan."
)


class Engine(Protocol):
    name: str

    def segments(self, audio_path: Path, tick: Tick = None) -> Segments: ...


class FasterWhisperEngine:
    """CPU Whisper. Works on any machine; much heavier than Parakeet on a Mac."""

    def __init__(self, model_size: str) -> None:
        from faster_whisper import WhisperModel

        self.name = f"whisper-{model_size}"
        # Leave cores free for the video call and the rest of the Mac.
        self._model = WhisperModel(
            model_size, device="cpu", compute_type="int8",
            cpu_threads=max(2, (os.cpu_count() or 4) // 2),
        )

    def segments(self, audio_path: Path, tick: Tick = None) -> Segments:
        stream, _ = self._model.transcribe(
            str(audio_path),
            language="en",
            beam_size=5,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 500},
            initial_prompt=_WHISPER_PROMPT,
            condition_on_previous_text=False,   # stops repetition loops
        )
        result: Segments = []
        for segment in stream:
            if tick:
                tick()
            text = segment.text.strip()
            if text:
                result.append((segment.start, text))
        return result


class ParakeetEngine:
    """Parakeet TDT v3 on the Apple GPU via MLX. Apple Silicon only."""

    CHUNK_SECONDS = 120.0
    OVERLAP_SECONDS = 15.0

    def __init__(self, model_id: str = PARAKEET_MODEL) -> None:
        from parakeet_mlx import from_pretrained, parakeet

        # parakeet-mlx decodes audio with an ffmpeg binary. Our recordings are
        # plain WAV files, so read them directly and avoid that system dependency.
        parakeet.load_audio = _load_wav_for_mlx
        self.name = "parakeet-v3"
        self._model = from_pretrained(model_id)

    def segments(self, audio_path: Path, tick: Tick = None) -> Segments:
        def on_chunk(_position, _total) -> None:
            if tick:
                tick()

        result = self._model.transcribe(
            str(audio_path),
            chunk_duration=self.CHUNK_SECONDS,
            overlap_duration=self.OVERLAP_SECONDS,
            chunk_callback=on_chunk,
        )
        return [(s.start, s.text.strip()) for s in result.sentences if s.text.strip()]


def _load_wav_for_mlx(path, sampling_rate, dtype=None):
    import mlx.core as mx

    with wave.open(str(path), "rb") as handle:
        if handle.getframerate() != sampling_rate or handle.getnchannels() != 1 or handle.getsampwidth() != 2:
            raise ValueError(f"Expected {sampling_rate} Hz mono 16-bit audio in {Path(path).name}")
        raw = handle.readframes(handle.getnframes())
    return mx.array(np.frombuffer(raw, np.int16).copy()).astype(mx.float32) / 32768.0


def create_engine(engine: str = "auto", whisper_model: str = "small") -> Engine:
    """Build the chosen engine. 'auto' falls back to Whisper if Parakeet can't load."""
    choice = recommend_models().asr_engine if engine == "auto" else engine
    if choice == "parakeet":
        try:
            return ParakeetEngine()
        except Exception as exc:  # noqa: BLE001 - missing package, no GPU, failed download
            if engine != "auto":
                raise
            log.warning("Parakeet unavailable (%s: %s); using Whisper", type(exc).__name__, exc)
    return FasterWhisperEngine(whisper_model)
