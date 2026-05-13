from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import scipy.io.wavfile as wavfile

SAMPLE_RATE = 16_000

# ---------------------------------------------------------------------------
# Shared transcript fixtures
# ---------------------------------------------------------------------------

SPEAKER_TRANSCRIPT = (
    "Therapist: How are you feeling today?\n\n"
    "Client: I've been anxious. My name is Sarah Johnson and I live in Portland.\n\n"
    "Therapist: Tell me more about the anxiety."
)


@pytest.fixture
def speaker_labeled_transcript() -> str:
    return SPEAKER_TRANSCRIPT


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------

def _make_wav(path: Path, duration_sec: float = 2.0, freq_hz: float = 440.0) -> Path:
    """Generate a short sine-wave WAV file at 16 kHz / int16."""
    t = np.linspace(0, duration_sec, int(SAMPLE_RATE * duration_sec), endpoint=False)
    tone = (0.4 * np.sin(2 * np.pi * freq_hz * t) * 32767).astype(np.int16)
    wavfile.write(str(path), SAMPLE_RATE, tone)
    return path


@pytest.fixture
def sample_wav(tmp_path: Path):
    """Factory fixture — call it with an optional filename to get a WAV path."""
    def _factory(name: str = "test.wav", duration: float = 2.0, freq: float = 440.0) -> Path:
        return _make_wav(tmp_path / name, duration, freq)
    return _factory


@pytest.fixture
def tmp_session_dir(tmp_path: Path) -> Path:
    d = tmp_path / "session"
    d.mkdir()
    return d


# ---------------------------------------------------------------------------
# Config helper — builds a Config with test-safe defaults
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_config(tmp_path: Path, monkeypatch):
    """Set env vars for a valid config, return a factory that creates Config().

    The factory allows tests to override individual env vars before construction.
    """
    template = tmp_path / "template.txt"
    template.write_text("THERAPY SESSION NOTE\n====================\nTest template.")

    defaults = {
        "AI_PROVIDER": "anthropic",
        "ANTHROPIC_API_KEY": "sk-test-key-fake",
        "OPENAI_API_KEY": "",
        "SESSION_DURATION_MINUTES": "55",
        "WHISPER_MODEL": "small",
        "SESSIONS_DIR": str(tmp_path / "sessions"),
        "TEMPLATE_FILE": str(template),
        "AUDIO_PRIMARY_DEVICE": "",
        "AUDIO_LOOPBACK_DEVICE": "",
        "OLLAMA_MODEL": "llama3.1",
        "OLLAMA_BASE_URL": "http://localhost:11434",
        "DELETE_AUDIO_AFTER_TRANSCRIPTION": "false",
        "DELETE_RAW_TRANSCRIPT": "false",
    }
    for k, v in defaults.items():
        monkeypatch.setenv(k, v)

    def _factory(**overrides):
        for k, v in overrides.items():
            monkeypatch.setenv(k, v)
        # Import here so env vars are read fresh each time
        from therapy_notes.config import Config
        return Config()

    return _factory
