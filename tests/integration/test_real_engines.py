"""Real speech engines, Presidio and the worker pipeline. Only the LLM is faked."""
from __future__ import annotations

import os
import platform
import wave
from pathlib import Path

import numpy as np
import pytest

from therapy_notes import pipeline, session as store
from therapy_notes.audio.recorder import SAMPLE_RATE

FIXTURES = Path(__file__).parent.parent / "fixtures"
apple_silicon = platform.system() == "Darwin" and platform.machine() == "arm64"

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def parakeet():
    if not apple_silicon:
        pytest.skip("Parakeet needs Apple Silicon")
    from therapy_notes.transcription.engines import ParakeetEngine
    return ParakeetEngine()


def test_parakeet_needs_no_ffmpeg(parakeet, monkeypatch):
    monkeypatch.setenv("PATH", "/nonexistent")
    segments = parakeet.segments(FIXTURES / "client_sample.wav")
    text = " ".join(t for _, t in segments).lower()
    assert "anxious" in text and "portland" in text


def test_parakeet_invents_nothing_from_silence(parakeet, tmp_path):
    path = tmp_path / "silence.wav"
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(SAMPLE_RATE)
        handle.writeframes(b"\0\0" * (SAMPLE_RATE * 150))
    assert parakeet.segments(path) == []


def test_parakeet_tick_allows_pausing_long_audio(parakeet, tmp_path):
    with wave.open(str(FIXTURES / "client_sample.wav")) as handle:
        frames = handle.readframes(handle.getnframes())
    path = tmp_path / "long.wav"
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(SAMPLE_RATE)
        handle.writeframes(frames * 60)
    ticks = []
    parakeet.segments(path, lambda: ticks.append(1))
    assert ticks


def test_whole_pipeline_keeps_pii_out_of_llm_input(tmp_path, monkeypatch, mock_config):
    import shutil

    monkeypatch.setattr(pipeline, "config", mock_config(ASR_ENGINE="auto", WHISPER_MODEL="tiny", DELETE_AUDIO_AFTER_NOTES="true"))
    session = store.create_session(tmp_path / "root", couples=False)
    shutil.copy(FIXTURES / "therapist_sample.wav", session / store.THERAPIST_AUDIO)
    shutil.copy(FIXTURES / "client_sample.wav", session / store.CLIENT_AUDIO)
    store.update_session(session, state=store.QUEUED)

    sent = {}
    backend = pipeline.LocalBackend()

    def fake_notes(text, couples, anonymized):
        sent.update(text=text, anonymized=anonymized)
        return "note"

    monkeypatch.setattr(backend, "generate_notes", fake_notes)
    pipeline.process_session(session, backend)

    assert "Therapist:" in sent["text"] and "Client:" in sent["text"]
    assert "Sarah Johnson" not in sent["text"] and "Portland" not in sent["text"]
    assert sent["anonymized"] is True
    assert store.load_session(session)["state"] == store.DONE
    assert store.audio_paths(session) == []
