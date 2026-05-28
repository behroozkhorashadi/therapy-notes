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


class TestPIITypes:
    """Different PII entity types are correctly detected and replaced."""

    def test_phone_number_replaced(self, anon: Anonymizer):
        text = "Client: My number is 555-123-4567 if you need to reach me."
        result = anon.anonymize(text)
        assert "555-123-4567" not in result
        assert "[PHONE]" in result

    def test_email_replaced(self, anon: Anonymizer):
        text = "Client: You can email me at jane.doe@example.com."
        result = anon.anonymize(text)
        assert "jane.doe@example.com" not in result
        assert "[EMAIL]" in result

    def test_multiple_pii_types_in_one_turn(self, anon: Anonymizer):
        text = (
            "Client: I'm Sarah Johnson. "
            "Reach me at sarah@example.com or 555-987-6543."
        )
        result = anon.anonymize(text)
        assert "Sarah Johnson" not in result
        assert "sarah@example.com" not in result
        assert "555-987-6543" not in result


class TestEdgeCases:
    """Boundary inputs that should not raise errors."""

    def test_empty_string(self, anon: Anonymizer):
        result = anon.anonymize("")
        assert result == ""

    def test_whitespace_only(self, anon: Anonymizer):
        result = anon.anonymize("   ")
        assert result.strip() == ""

    def test_pii_immediately_after_label(self, anon: Anonymizer):
        """Name right after 'Client: ' with no leading text must still be caught."""
        text = "Client: John Smith said he was feeling better."
        result = anon.anonymize(text)
        assert "Client: " in result
        assert "John Smith" not in result

    def test_load_is_idempotent(self, anon: Anonymizer):
        """Calling _load() multiple times returns the same engine objects."""
        engines_a = anon._load()
        engines_b = anon._load()
        assert engines_a is engines_b
