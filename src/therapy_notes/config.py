"""Application settings, read from .env and the environment.

All relative paths resolve against the repo root, so the app behaves the same
whichever folder it is launched from.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

from .models import recommend_models

REPO_ROOT = Path(__file__).resolve().parents[2]
ENV_PATH = REPO_ROOT / ".env"

load_dotenv(ENV_PATH)


def _s(key: str, default: str = "") -> str:
    return os.getenv(key, default)


def _i(key: str, default: int) -> int:
    try:
        return int(os.getenv(key, str(default)))
    except ValueError:
        return default


def _b(key: str, default: bool = False) -> bool:
    return os.getenv(key, str(default)).lower() in ("1", "true", "yes")


def _opt(key: str) -> Optional[str]:
    val = os.getenv(key, "").strip()
    return val or None


def _path(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def data_dir() -> Path:
    """Folder for logs and, on new installs, sessions."""
    override = os.getenv("THERAPY_NOTES_DATA_DIR", "").strip()
    if override:
        return _path(override)
    return Path.home() / "Library" / "Application Support" / "TherapyNotes"


def _default_sessions_dir() -> str:
    # Existing installs keep their sessions in the repo folder until moved.
    legacy = REPO_ROOT / "sessions"
    return str(legacy if legacy.exists() else data_dir() / "sessions")


_VALID_WHISPER_MODELS = {
    "tiny", "tiny.en",
    "base", "base.en",
    "small", "small.en",
    "medium", "medium.en",
    "large-v1", "large-v2", "large-v3", "large-v3-turbo", "turbo",
    "distil-large-v2", "distil-large-v3", "distil-large-v3.5",
}
_VALID_ASR_ENGINES = {"auto", "parakeet", "faster-whisper"}
_PLACEHOLDER_KEYS = {"your_anthropic_key_here", "your_openai_key_here"}


def _key_ok(key: str) -> bool:
    return bool(key.strip()) and key.strip() not in _PLACEHOLDER_KEYS


@dataclass
class Config:
    # ── AI provider ───────────────────────────────────────────────────────────
    ai_provider: str = field(default_factory=lambda: _s("AI_PROVIDER", "anthropic"))
    anthropic_api_key: str = field(default_factory=lambda: _s("ANTHROPIC_API_KEY"))
    openai_api_key: str = field(default_factory=lambda: _s("OPENAI_API_KEY"))
    ai_model_anthropic: str = field(default_factory=lambda: _s("AI_MODEL_ANTHROPIC", "claude-opus-5-5"))
    ai_model_openai: str = field(default_factory=lambda: _s("AI_MODEL_OPENAI", "gpt-4o"))
    ollama_model: str = field(default_factory=lambda: _s("OLLAMA_MODEL", recommend_models().local_llm or "qwen3.5:9b"))
    ollama_base_url: str = field(default_factory=lambda: _s("OLLAMA_BASE_URL", "http://localhost:11434"))

    # ── Session ───────────────────────────────────────────────────────────────
    session_duration_minutes: int = field(default_factory=lambda: _i("SESSION_DURATION_MINUTES", 55))

    # ── Transcription ─────────────────────────────────────────────────────────
    asr_engine: str = field(default_factory=lambda: _s("ASR_ENGINE", "auto"))
    whisper_model: str = field(default_factory=lambda: _s("WHISPER_MODEL", recommend_models().whisper_model))

    # ── Audio devices (None = system default) ─────────────────────────────────
    audio_primary_device: Optional[str] = field(default_factory=lambda: _opt("AUDIO_PRIMARY_DEVICE"))
    audio_loopback_device: Optional[str] = field(default_factory=lambda: _opt("AUDIO_LOOPBACK_DEVICE"))

    # ── Paths ─────────────────────────────────────────────────────────────────
    sessions_dir: Path = field(default_factory=lambda: _path(_s("SESSIONS_DIR", _default_sessions_dir())))
    template_file: Path = field(default_factory=lambda: _path(_s("TEMPLATE_FILE", "templates/notes_template.txt")))
    couples_template_file: Path = field(default_factory=lambda: _path(_s("COUPLES_TEMPLATE_FILE", "templates/notes_template_couples.txt")))

    # ── Privacy ───────────────────────────────────────────────────────────────
    # Audio is only ever deleted after notes have been saved.
    delete_audio_after_notes: bool = field(default_factory=lambda: _b("DELETE_AUDIO_AFTER_NOTES", True))
    delete_raw_transcript: bool = field(default_factory=lambda: _b("DELETE_RAW_TRANSCRIPT"))

    def validate(self) -> list[str]:
        errors: list[str] = []
        if self.ai_provider not in ("anthropic", "openai", "ollama"):
            errors.append(f"AI_PROVIDER must be 'anthropic', 'openai', or 'ollama' — got '{self.ai_provider}'")
        if self.ai_provider == "anthropic" and not _key_ok(self.anthropic_api_key):
            errors.append("ANTHROPIC_API_KEY is not set in .env")
        if self.ai_provider == "openai" and not _key_ok(self.openai_api_key):
            errors.append("OPENAI_API_KEY is not set in .env")
        if self.ai_provider == "ollama" and not self.ollama_model:
            errors.append("OLLAMA_MODEL is not set in .env")
        if self.session_duration_minutes <= 0:
            errors.append("SESSION_DURATION_MINUTES must be a positive integer")
        if self.asr_engine not in _VALID_ASR_ENGINES:
            errors.append(f"ASR_ENGINE must be one of {sorted(_VALID_ASR_ENGINES)} — got '{self.asr_engine}'")
        if self.whisper_model not in _VALID_WHISPER_MODELS:
            errors.append(f"WHISPER_MODEL '{self.whisper_model}' is not a recognized model size")
        if not self.template_file.exists():
            errors.append(f"Note template not found: {self.template_file}")
        return errors


config = Config()


def reload_config() -> Config:
    """Re-read .env and update the shared config object in place."""
    load_dotenv(ENV_PATH, override=True)
    config.__dict__.update(Config().__dict__)
    return config
