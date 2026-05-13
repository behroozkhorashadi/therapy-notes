"""Integration tests that exercise the real transcription and anonymization pipeline.

These tests load the Whisper "tiny" model and run real Presidio engines.
They are marked @pytest.mark.slow so they can be skipped in fast CI runs.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"
THERAPIST_WAV = FIXTURES_DIR / "therapist_sample.wav"
CLIENT_WAV = FIXTURES_DIR / "client_sample.wav"

# Skip the entire module if fixture WAVs are missing
pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        not THERAPIST_WAV.exists() or not CLIENT_WAV.exists(),
        reason="Fixture WAVs not found — run `python tests/generate_fixtures.py`",
    ),
]


@pytest.fixture(scope="module")
def transcriber():
    """Load Whisper tiny once for the whole module (expensive)."""
    from therapy_notes.transcription.transcriber import Transcriber
    t = Transcriber(model_size="tiny")
    t._load()  # force load so timing is in fixture, not first test
    return t


@pytest.fixture(scope="module")
def anonymizer():
    from therapy_notes.anonymization.anonymizer import Anonymizer
    a = Anonymizer()
    a._load()
    return a


class TestFullPipeline:
    """Fixture WAVs -> transcribe -> anonymize -> verify."""

    def test_two_speaker_transcription(self, transcriber):
        result = transcriber.transcribe_two_speakers(THERAPIST_WAV, CLIENT_WAV)
        assert len(result) > 0
        # Should contain speaker labels
        assert "Therapist:" in result or "Client:" in result

    def test_two_speaker_has_content(self, transcriber):
        result = transcriber.transcribe_two_speakers(THERAPIST_WAV, CLIENT_WAV)
        # Whisper tiny may not produce exact words, but it should
        # produce *something* from both tracks
        lines = result.split("\n\n")
        assert len(lines) >= 2, "Expected at least two speaker turns"

    def test_anonymize_preserves_labels(self, transcriber, anonymizer):
        transcript = transcriber.transcribe_two_speakers(THERAPIST_WAV, CLIENT_WAV)
        anonymized = anonymizer.anonymize(transcript)
        # Labels should survive
        if "Therapist:" in transcript:
            assert "Therapist:" in anonymized
        if "Client:" in transcript:
            assert "Client:" in anonymized


class TestSingleTrackFallback:
    """Single audio file -> flat transcription (no labels)."""

    def test_single_track_produces_text(self, transcriber):
        result = transcriber.transcribe(THERAPIST_WAV)
        assert len(result) > 0
        # Single track should NOT have speaker labels
        assert "Therapist:" not in result
        assert "Client:" not in result


class TestAnonymizerOnRealisticTranscript:
    """Hardcoded realistic transcript -> anonymize -> verify."""

    def test_labels_intact_pii_replaced(self, anonymizer):
        transcript = (
            "Therapist: How have you been this week?\n\n"
            "Client: Not great. I talked to my sister Emily Davis in Chicago "
            "and she made me feel worse. My phone number is 555-123-4567.\n\n"
            "Therapist: That sounds difficult. Let's explore that."
        )
        result = anonymizer.anonymize(transcript)

        # Labels must survive
        assert result.count("Therapist: ") == 2
        assert result.count("Client: ") == 1

        # PII must be gone
        assert "Emily Davis" not in result
        assert "Chicago" not in result
        assert "555-123-4567" not in result

        # Placeholders should be present
        assert "[PERSON]" in result

    def test_non_pii_content_preserved(self, anonymizer):
        transcript = (
            "Therapist: Tell me about your anxiety.\n\n"
            "Client: I feel anxious at work. No specific triggers."
        )
        result = anonymizer.anonymize(transcript)
        # Content without PII should pass through mostly unchanged
        assert "anxiety" in result.lower() or "anxious" in result.lower()
        assert "work" in result
