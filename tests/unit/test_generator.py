"""Tests for the NoteGenerator prompt construction and routing.

These tests mock API calls — they verify prompt shape and provider routing,
not the actual API responses.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from therapy_notes.notes.generator import NoteGenerator, _SYSTEM_PROMPT


class TestSystemPrompt:
    """The system prompt should contain key instructions."""

    def test_mentions_therapist_label(self):
        assert "Therapist:" in _SYSTEM_PROMPT

    def test_mentions_client_label(self):
        assert "Client:" in _SYSTEM_PROMPT

    def test_mentions_pii_placeholders(self):
        assert "[PERSON]" in _SYSTEM_PROMPT

    def test_mentions_soap(self):
        assert "SOAP" in _SYSTEM_PROMPT

    def test_mentions_subjective_objective_attribution(self):
        assert "Subjective" in _SYSTEM_PROMPT
        assert "Objective" in _SYSTEM_PROMPT


class TestPromptConstruction:
    """generate() should build a prompt with transcript and template tags."""

    @pytest.fixture
    def generator(self, tmp_path: Path, monkeypatch):
        template = tmp_path / "template.txt"
        template.write_text("SESSION NOTE\n============\nS:\nO:\nA:\nP:")
        monkeypatch.setenv("TEMPLATE_FILE", str(template))
        monkeypatch.setenv("AI_PROVIDER", "anthropic")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        from therapy_notes.config import Config
        cfg = Config()
        # Keep the patch active for the entire test
        with patch("therapy_notes.notes.generator.config", cfg):
            gen = NoteGenerator()
            yield gen

    def test_prompt_has_transcript_tags(self, generator):
        transcript = "Therapist: Hello\n\nClient: Hi"
        with patch.object(generator, "_call_anthropic", return_value="notes") as mock:
            generator.generate(transcript)
        prompt = mock.call_args[0][0]
        assert "<transcript>" in prompt
        assert "</transcript>" in prompt
        assert transcript in prompt

    def test_prompt_has_template_tags(self, generator):
        with patch.object(generator, "_call_anthropic", return_value="notes") as mock:
            generator.generate("some transcript")
        prompt = mock.call_args[0][0]
        assert "<template>" in prompt
        assert "</template>" in prompt


class TestProviderRouting:
    """generate() should dispatch to the correct provider method."""

    @pytest.fixture
    def make_generator(self, tmp_path: Path, monkeypatch):
        template = tmp_path / "template.txt"
        template.write_text("TEMPLATE")

        def _factory(provider: str):
            monkeypatch.setenv("TEMPLATE_FILE", str(template))
            monkeypatch.setenv("AI_PROVIDER", provider)
            monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
            monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
            monkeypatch.setenv("OLLAMA_MODEL", "llama3.1")
            from therapy_notes.config import Config
            cfg = Config()
            patcher = patch("therapy_notes.notes.generator.config", cfg)
            patcher.start()
            gen = NoteGenerator()
            return gen, patcher
        return _factory

    def test_anthropic_routing(self, make_generator):
        gen, patcher = make_generator("anthropic")
        try:
            with patch.object(gen, "_call_anthropic", return_value="notes") as mock:
                gen.generate("transcript")
            mock.assert_called_once()
        finally:
            patcher.stop()

    def test_openai_routing(self, make_generator):
        gen, patcher = make_generator("openai")
        try:
            with patch.object(gen, "_call_openai", return_value="notes") as mock:
                gen.generate("transcript")
            mock.assert_called_once()
        finally:
            patcher.stop()

    def test_ollama_routing(self, make_generator):
        gen, patcher = make_generator("ollama")
        try:
            with patch.object(gen, "_call_ollama", return_value="notes") as mock:
                gen.generate("transcript")
            mock.assert_called_once()
        finally:
            patcher.stop()
