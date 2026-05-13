"""Tests for Config validation logic."""
from __future__ import annotations

import pytest


class TestValidConfig:
    def test_valid_anthropic_config(self, mock_config):
        cfg = mock_config()
        errors = cfg.validate()
        assert errors == []

    def test_valid_openai_config(self, mock_config):
        cfg = mock_config(AI_PROVIDER="openai", OPENAI_API_KEY="sk-test-openai")
        errors = cfg.validate()
        assert errors == []

    def test_valid_ollama_config(self, mock_config):
        cfg = mock_config(AI_PROVIDER="ollama", OLLAMA_MODEL="llama3.1")
        errors = cfg.validate()
        assert errors == []


class TestProviderValidation:
    def test_invalid_provider(self, mock_config):
        cfg = mock_config(AI_PROVIDER="gemini")
        errors = cfg.validate()
        assert any("AI_PROVIDER" in e for e in errors)

    def test_missing_anthropic_key(self, mock_config):
        cfg = mock_config(AI_PROVIDER="anthropic", ANTHROPIC_API_KEY="")
        errors = cfg.validate()
        assert any("ANTHROPIC_API_KEY" in e for e in errors)

    def test_missing_openai_key(self, mock_config):
        cfg = mock_config(AI_PROVIDER="openai", OPENAI_API_KEY="")
        errors = cfg.validate()
        assert any("OPENAI_API_KEY" in e for e in errors)

    def test_ollama_no_key_needed(self, mock_config):
        cfg = mock_config(AI_PROVIDER="ollama")
        errors = cfg.validate()
        # Should not ask for any API key
        assert not any("API_KEY" in e for e in errors)

    def test_ollama_missing_model(self, mock_config):
        cfg = mock_config(AI_PROVIDER="ollama", OLLAMA_MODEL="")
        errors = cfg.validate()
        assert any("OLLAMA_MODEL" in e for e in errors)


class TestFieldValidation:
    def test_duration_zero(self, mock_config):
        cfg = mock_config(SESSION_DURATION_MINUTES="0")
        errors = cfg.validate()
        assert any("SESSION_DURATION_MINUTES" in e for e in errors)

    def test_duration_negative(self, mock_config):
        cfg = mock_config(SESSION_DURATION_MINUTES="-5")
        errors = cfg.validate()
        assert any("SESSION_DURATION_MINUTES" in e for e in errors)

    def test_invalid_whisper_model(self, mock_config):
        cfg = mock_config(WHISPER_MODEL="fake-model")
        errors = cfg.validate()
        assert any("WHISPER_MODEL" in e for e in errors)

    def test_valid_whisper_models(self, mock_config):
        for model in ("tiny", "base", "small", "medium", "large-v3", "distil-large-v3"):
            cfg = mock_config(WHISPER_MODEL=model)
            errors = cfg.validate()
            assert not any("WHISPER_MODEL" in e for e in errors), f"{model} should be valid"

    def test_missing_template(self, mock_config):
        cfg = mock_config(TEMPLATE_FILE="/nonexistent/template.txt")
        errors = cfg.validate()
        assert any("template" in e.lower() for e in errors)
