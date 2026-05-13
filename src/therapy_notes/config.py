from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv()


def _s(key: str, default: str = "") -> str:
    return os.getenv(key, default)


def _i(key: str, default: int) -> int:
    return int(os.getenv(key, str(default)))


def _b(key: str, default: bool = False) -> bool:
    return os.getenv(key, str(default)).lower() in ("1", "true", "yes")


def _opt(key: str) -> Optional[str]:
    val = os.getenv(key, "").strip()
    return val or None


_VALID_WHISPER_MODELS = {
    "tiny", "tiny.en",
    "base", "base.en",
    "small", "small.en",
    "medium", "medium.en",
    "large-v1", "large-v2", "large-v3",
    "distil-large-v2", "distil-large-v3",
}


@dataclass
class Config:
    # ── AI provider ───────────────────────────────────────────────────────────
    ai_provider: str = field(default_factory=lambda: _s("AI_PROVIDER", "anthropic"))
    anthropic_api_key: str = field(default_factory=lambda: _s("ANTHROPIC_API_KEY"))
    openai_api_key: str = field(default_factory=lambda: _s("OPENAI_API_KEY"))
    ai_model_anthropic: str = field(default_factory=lambda: _s("AI_MODEL_ANTHROPIC", "claude-sonnet-4-6"))
    ai_model_openai: str = field(default_factory=lambda: _s("AI_MODEL_OPENAI", "gpt-4o"))
    ollama_model: str = field(default_factory=lambda: _s("OLLAMA_MODEL", "llama3.1"))
    ollama_base_url: str = field(default_factory=lambda: _s("OLLAMA_BASE_URL", "http://localhost:11434"))

    # ── Session ───────────────────────────────────────────────────────────────
    session_duration_minutes: int = field(default_factory=lambda: _i("SESSION_DURATION_MINUTES", 55))

    # ── Transcription ─────────────────────────────────────────────────────────
    whisper_model: str = field(default_factory=lambda: _s("WHISPER_MODEL", "small"))

    # ── Audio devices (None = system default) ─────────────────────────────────
    audio_primary_device: Optional[str] = field(default_factory=lambda: _opt("AUDIO_PRIMARY_DEVICE"))
    audio_loopback_device: Optional[str] = field(default_factory=lambda: _opt("AUDIO_LOOPBACK_DEVICE"))

    # ── Paths ─────────────────────────────────────────────────────────────────
    sessions_dir: Path = field(default_factory=lambda: Path(_s("SESSIONS_DIR", "./sessions")))
    template_file: Path = field(default_factory=lambda: Path(_s("TEMPLATE_FILE", "./templates/notes_template.txt")))
    couples_template_file: Path = field(default_factory=lambda: Path(_s("COUPLES_TEMPLATE_FILE", "./templates/notes_template_couples.txt")))

    # ── Privacy ───────────────────────────────────────────────────────────────
    delete_audio_after_transcription: bool = field(default_factory=lambda: _b("DELETE_AUDIO_AFTER_TRANSCRIPTION"))
    delete_raw_transcript: bool = field(default_factory=lambda: _b("DELETE_RAW_TRANSCRIPT"))

    def validate(self) -> list[str]:
        errors: list[str] = []
        if self.ai_provider not in ("anthropic", "openai", "ollama"):
            errors.append(f"AI_PROVIDER must be 'anthropic', 'openai', or 'ollama' — got '{self.ai_provider}'")
        if self.ai_provider == "anthropic" and not self.anthropic_api_key:
            errors.append("ANTHROPIC_API_KEY is not set in .env")
        if self.ai_provider == "openai" and not self.openai_api_key:
            errors.append("OPENAI_API_KEY is not set in .env")
        if self.ai_provider == "ollama" and not self.ollama_model:
            errors.append("OLLAMA_MODEL is not set in .env")
        if self.session_duration_minutes <= 0:
            errors.append("SESSION_DURATION_MINUTES must be a positive integer")
        if self.whisper_model not in _VALID_WHISPER_MODELS:
            errors.append(f"WHISPER_MODEL '{self.whisper_model}' is not a recognized model size")
        if not self.template_file.exists():
            errors.append(f"Note template not found: {self.template_file}")
        return errors


config = Config()
