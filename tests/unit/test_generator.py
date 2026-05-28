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


class TestMissingTemplate:
    """NoteGenerator raises FileNotFoundError when a template path doesn't exist."""

    def test_missing_individual_template(self, monkeypatch):
        monkeypatch.setenv("TEMPLATE_FILE", "/nonexistent/template.txt")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        from therapy_notes.config import Config
        cfg = Config()
        with patch("therapy_notes.notes.generator.config", cfg):
            with pytest.raises(FileNotFoundError, match="Template not found"):
                NoteGenerator(couples=False)

    def test_missing_couples_template(self, tmp_path, monkeypatch):
        # Individual template exists but couples template does not
        individual = tmp_path / "template.txt"
        individual.write_text("TEMPLATE")
        monkeypatch.setenv("TEMPLATE_FILE", str(individual))
        monkeypatch.setenv("COUPLES_TEMPLATE_FILE", "/nonexistent/couples.txt")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        from therapy_notes.config import Config
        cfg = Config()
        with patch("therapy_notes.notes.generator.config", cfg):
            with pytest.raises(FileNotFoundError, match="Template not found"):
                NoteGenerator(couples=True)


class TestAnonymizedFlag:
    """The prompt label changes based on the anonymized flag."""

    @pytest.fixture
    def make_gen(self, tmp_path, monkeypatch):
        template = tmp_path / "template.txt"
        template.write_text("S:\nO:\nA:\nP:")
        monkeypatch.setenv("TEMPLATE_FILE", str(template))
        monkeypatch.setenv("AI_PROVIDER", "anthropic")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        from therapy_notes.config import Config
        cfg = Config()

        def _factory(anonymized: bool):
            with patch("therapy_notes.notes.generator.config", cfg):
                return NoteGenerator(anonymized=anonymized), cfg
        return _factory

    def test_anonymized_prompt_label(self, make_gen):
        gen, cfg = make_gen(anonymized=True)
        with patch("therapy_notes.notes.generator.config", cfg):
            with patch.object(gen, "_call_anthropic", return_value="notes") as mock:
                gen.generate("transcript")
        prompt = mock.call_args[0][0]
        assert prompt.startswith("Anonymized session transcript")

    def test_not_anonymized_prompt_label(self, make_gen):
        gen, cfg = make_gen(anonymized=False)
        with patch("therapy_notes.notes.generator.config", cfg):
            with patch.object(gen, "_call_anthropic", return_value="notes") as mock:
                gen.generate("transcript")
        prompt = mock.call_args[0][0]
        assert prompt.startswith("Session transcript")
        assert not prompt.startswith("Anonymized")


class TestCouplesMode:
    """NoteGenerator with couples=True uses the couples template and prompt."""

    @pytest.fixture
    def couples_gen(self, tmp_path, monkeypatch):
        individual = tmp_path / "template.txt"
        individual.write_text("INDIVIDUAL TEMPLATE")
        couples = tmp_path / "couples_template.txt"
        couples.write_text("COUPLES TEMPLATE")
        monkeypatch.setenv("TEMPLATE_FILE", str(individual))
        monkeypatch.setenv("COUPLES_TEMPLATE_FILE", str(couples))
        monkeypatch.setenv("AI_PROVIDER", "anthropic")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        from therapy_notes.config import Config
        cfg = Config()
        with patch("therapy_notes.notes.generator.config", cfg):
            gen = NoteGenerator(couples=True)
            yield gen, cfg

    def test_couples_template_in_prompt(self, couples_gen):
        gen, cfg = couples_gen
        with patch("therapy_notes.notes.generator.config", cfg):
            with patch.object(gen, "_call_anthropic", return_value="notes") as mock:
                gen.generate("transcript")
        prompt = mock.call_args[0][0]
        assert "COUPLES TEMPLATE" in prompt
        assert "INDIVIDUAL TEMPLATE" not in prompt

    def test_couples_system_prompt_content(self):
        from therapy_notes.notes.generator import _build_system_prompt
        prompt = _build_system_prompt(couples=True, anonymized=True)
        assert "couples" in prompt.lower()
        # Couples body explains the shared Client: channel
        assert "Partner" in prompt

    def test_individual_system_prompt_content(self):
        from therapy_notes.notes.generator import _build_system_prompt
        prompt = _build_system_prompt(couples=False, anonymized=True)
        assert "SOAP" in prompt
        assert "Subjective" in prompt
        assert "Objective" in prompt
        # Individual prompt should NOT mention couples-specific partner logic
        assert "Partner 1" not in prompt
