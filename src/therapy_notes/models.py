"""Which local models to use on this machine.

Keep this catalog current: it is the one place to update when better local
models ship. Ollama tags were checked against ollama.com/library on 2026-10-31.
Picks for local note writing have not been evaluated on clinical sessions.
"""
from __future__ import annotations

import os
import platform
from dataclasses import dataclass
from typing import Optional

PARAKEET_MODEL = "mlx-community/parakeet-tdt-0.6b-v3"

# (minimum RAM in GB, Ollama tag, approximate download size). Each tag must leave
# room for macOS, the speech engine and a large context window. Largest first.
_LOCAL_LLM_TIERS = (
    (32, "qwen3.8:27b", "18 GB"),
    (24, "gpt-oss:20b", "14 GB"),
    (16, "qwen3.5:9b", "6.6–7.6 GB"),
)


@dataclass(frozen=True)
class Hardware:
    ram_gb: int
    apple_silicon: bool


@dataclass(frozen=True)
class Recommendation:
    asr_engine: str                # "parakeet" or "faster-whisper"
    whisper_model: str             # fallback engine's model
    local_llm: Optional[str]       # None: this machine should use a cloud provider
    local_llm_size: Optional[str]
    summary: str


def detect_hardware() -> Hardware:
    try:
        ram = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError):
        ram = 8 * 1024**3
    apple = platform.system() == "Darwin" and platform.machine() == "arm64"
    return Hardware(ram_gb=round(ram / 1024**3), apple_silicon=apple)


def recommend_models(hw: Optional[Hardware] = None) -> Recommendation:
    hw = hw or detect_hardware()
    whisper = "large-v3-turbo" if hw.ram_gb >= 16 else "small"
    engine = "parakeet" if hw.apple_silicon else "faster-whisper"
    llm, size = None, None
    for min_ram, tag, tag_size in _LOCAL_LLM_TIERS:
        if hw.ram_gb >= min_ram:
            llm, size = tag, tag_size
            break
    speech = "Parakeet v3" if engine == "parakeet" else f"Whisper {whisper}"
    notes = (
        f"{llm} via Ollama ({size} download)"
        if llm
        else "a cloud provider, because this Mac has too little memory for a good local model"
    )
    return Recommendation(engine, whisper, llm, size, f"{hw.ram_gb} GB Mac: {speech} for speech, {notes} for notes.")
