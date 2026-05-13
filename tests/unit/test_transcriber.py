"""Tests for the Transcriber's two-speaker interleaving logic.

These tests mock _get_timed_segments to avoid loading Whisper.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from therapy_notes.transcription.transcriber import Transcriber


@pytest.fixture
def transcriber() -> Transcriber:
    return Transcriber(model_size="tiny")


DUMMY = Path("/dummy.wav")


class TestTwoSpeakerInterleaving:
    """transcribe_two_speakers merges two segment lists by timestamp."""

    def test_basic_interleaving(self, transcriber: Transcriber):
        therapist_segs = [(0.0, "How are you")]
        client_segs = [(2.0, "Anxious")]
        with patch.object(transcriber, "_get_timed_segments", side_effect=[therapist_segs, client_segs]):
            result = transcriber.transcribe_two_speakers(DUMMY, DUMMY)
        assert result == "Therapist: How are you\n\nClient: Anxious"

    def test_consecutive_merge(self, transcriber: Transcriber):
        """Consecutive segments from the same speaker merge into one line."""
        therapist_segs = [(0.0, "Let me"), (0.5, "ask you")]
        client_segs = [(3.0, "Sure")]
        with patch.object(transcriber, "_get_timed_segments", side_effect=[therapist_segs, client_segs]):
            result = transcriber.transcribe_two_speakers(DUMMY, DUMMY)
        assert result == "Therapist: Let me ask you\n\nClient: Sure"

    def test_single_speaker(self, transcriber: Transcriber):
        """Only therapist has audio — output should be therapist only."""
        therapist_segs = [(0.0, "Hello")]
        client_segs = []
        with patch.object(transcriber, "_get_timed_segments", side_effect=[therapist_segs, client_segs]):
            result = transcriber.transcribe_two_speakers(DUMMY, DUMMY)
        assert result == "Therapist: Hello"

    def test_timestamp_sorting(self, transcriber: Transcriber):
        """Segments should be sorted by timestamp regardless of source order."""
        therapist_segs = [(5.0, "Tell me more")]
        client_segs = [(1.0, "I feel anxious"), (8.0, "It started last week")]
        with patch.object(transcriber, "_get_timed_segments", side_effect=[therapist_segs, client_segs]):
            result = transcriber.transcribe_two_speakers(DUMMY, DUMMY)
        lines = result.split("\n\n")
        assert lines[0].startswith("Client:")
        assert lines[1].startswith("Therapist:")
        assert lines[2].startswith("Client:")

    def test_both_empty(self, transcriber: Transcriber):
        """No segments from either speaker → empty string."""
        with patch.object(transcriber, "_get_timed_segments", side_effect=[[], []]):
            result = transcriber.transcribe_two_speakers(DUMMY, DUMMY)
        assert result == ""

    def test_multiple_alternations(self, transcriber: Transcriber):
        """Full back-and-forth conversation."""
        therapist_segs = [
            (0.0, "Hi there"),
            (4.0, "How has your week been"),
        ]
        client_segs = [
            (2.0, "Hello"),
            (6.0, "Pretty tough actually"),
        ]
        with patch.object(transcriber, "_get_timed_segments", side_effect=[therapist_segs, client_segs]):
            result = transcriber.transcribe_two_speakers(DUMMY, DUMMY)
        lines = result.split("\n\n")
        assert len(lines) == 4
        assert lines[0] == "Therapist: Hi there"
        assert lines[1] == "Client: Hello"
        assert lines[2] == "Therapist: How has your week been"
        assert lines[3] == "Client: Pretty tough actually"
