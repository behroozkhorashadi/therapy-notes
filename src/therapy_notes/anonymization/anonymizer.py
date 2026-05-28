from __future__ import annotations

"""
Local PII detection and redaction using Microsoft Presidio.
No data leaves the machine during this step.

Requires the spaCy model (download once):
    python -m spacy download en_core_web_lg

Each detected entity is replaced with a clearly-labeled placeholder so the
AI note generator understands context without seeing real names or dates.
"""

import re

from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig

_ENTITIES = [
    "PERSON",
    "PHONE_NUMBER",
    "EMAIL_ADDRESS",
    "LOCATION",
    "DATE_TIME",
    "US_SSN",
    "MEDICAL_LICENSE",
    "NRP",          # nationality, religion, political group
    "CREDIT_CARD",
    "IP_ADDRESS",
    "US_DRIVER_LICENSE",
    "US_PASSPORT",
]

# Descriptive placeholders so the AI can still reason about the context
_OPERATORS: dict[str, OperatorConfig] = {
    "PERSON":            OperatorConfig("replace", {"new_value": "[PERSON]"}),
    "PHONE_NUMBER":      OperatorConfig("replace", {"new_value": "[PHONE]"}),
    "EMAIL_ADDRESS":     OperatorConfig("replace", {"new_value": "[EMAIL]"}),
    "LOCATION":          OperatorConfig("replace", {"new_value": "[LOCATION]"}),
    "DATE_TIME":         OperatorConfig("replace", {"new_value": "[DATE]"}),
    "US_SSN":            OperatorConfig("replace", {"new_value": "[SSN]"}),
    "MEDICAL_LICENSE":   OperatorConfig("replace", {"new_value": "[LICENSE]"}),
    "NRP":               OperatorConfig("replace", {"new_value": "[NRP]"}),
    "CREDIT_CARD":       OperatorConfig("replace", {"new_value": "[CC]"}),
    "IP_ADDRESS":        OperatorConfig("replace", {"new_value": "[IP]"}),
    "US_DRIVER_LICENSE": OperatorConfig("replace", {"new_value": "[DL]"}),
    "US_PASSPORT":       OperatorConfig("replace", {"new_value": "[PASSPORT]"}),
}


_SPEAKER_RE = re.compile(r"((?:Therapist|Client): )")


class Anonymizer:
    def __init__(self) -> None:
        # Engines are initialized lazily on first use (each ~1–2 s to load).
        self._engines: tuple[AnalyzerEngine, AnonymizerEngine] | None = None

    def _load(self) -> tuple[AnalyzerEngine, AnonymizerEngine]:
        """Lazy-initialize Presidio engines and return them."""
        if self._engines is None:
            self._engines = (AnalyzerEngine(), AnonymizerEngine())
        return self._engines

    def anonymize(self, text: str) -> str:
        """Remove PII from text and return the redacted version.

        Speaker labels ("Therapist: " / "Client: ") are preserved by
        anonymizing each speaker turn separately.
        """
        parts = _SPEAKER_RE.split(text)
        if len(parts) == 1:
            # No speaker labels — flat transcript, anonymize whole text
            return self._anonymize_chunk(text)
        # parts = ['', 'Therapist: ', 'body...', 'Client: ', 'body...', ...]
        # Anonymize only the body segments (indices 2, 4, 6, ...)
        for i in range(2, len(parts), 2):
            parts[i] = self._anonymize_chunk(parts[i])
        return "".join(parts)

    def _anonymize_chunk(self, text: str) -> str:
        analyzer, anonymizer = self._load()
        results = analyzer.analyze(
            text=text,
            language="en",
            entities=_ENTITIES,
            score_threshold=0.4,
        )
        result = anonymizer.anonymize(
            text=text,
            analyzer_results=results,
            operators=_OPERATORS,
        )
        return result.text
