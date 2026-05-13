"""Tests for the Presidio-based anonymizer.

These tests run against real Presidio engines (not mocked) to verify
that speaker labels survive anonymization and PII is correctly replaced.
"""
from __future__ import annotations

import pytest

from therapy_notes.anonymization.anonymizer import Anonymizer


@pytest.fixture
def anon() -> Anonymizer:
    return Anonymizer()


class TestSpeakerLabelsSurvive:
    """Speaker prefixes must pass through anonymization unchanged."""

    def test_labels_intact(self, anon: Anonymizer, speaker_labeled_transcript: str):
        result = anon.anonymize(speaker_labeled_transcript)
        assert "Therapist: " in result
        assert "Client: " in result

    def test_label_count_preserved(self, anon: Anonymizer):
        text = (
            "Therapist: Hello there.\n\n"
            "Client: Hi.\n\n"
            "Therapist: How are you?"
        )
        result = anon.anonymize(text)
        assert result.count("Therapist: ") == 2
        assert result.count("Client: ") == 1


class TestPIIReplacement:
    """PII within speaker turns should be replaced with placeholders."""

    def test_person_name_replaced(self, anon: Anonymizer, speaker_labeled_transcript: str):
        result = anon.anonymize(speaker_labeled_transcript)
        assert "Sarah Johnson" not in result
        assert "[PERSON]" in result

    def test_location_replaced(self, anon: Anonymizer, speaker_labeled_transcript: str):
        result = anon.anonymize(speaker_labeled_transcript)
        assert "Portland" not in result
        assert "[LOCATION]" in result

    def test_pii_right_after_label(self, anon: Anonymizer):
        text = "Client: John Smith felt sad about moving."
        result = anon.anonymize(text)
        assert "Client: " in result
        assert "John Smith" not in result
        assert "[PERSON]" in result


class TestNoLabels:
    """Flat text without speaker labels should still be anonymized."""

    def test_no_pii_passthrough(self, anon: Anonymizer):
        text = "The weather was cloudy and the session went well."
        result = anon.anonymize(text)
        assert result == text

    def test_flat_text_pii_replaced(self, anon: Anonymizer):
        text = "The client Sarah Johnson from Portland discussed her anxiety."
        result = anon.anonymize(text)
        assert "Sarah Johnson" not in result
        assert "Portland" not in result
        assert "[PERSON]" in result
        assert "[LOCATION]" in result
