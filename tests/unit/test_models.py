"""The model catalog picks something that fits each machine."""
from __future__ import annotations

import pytest

from therapy_notes.models import Hardware, recommend_models


@pytest.mark.parametrize("ram,llm", [
    (8, None), (15, None), (16, "qwen3.5:9b"), (24, "gpt-oss:20b"), (32, "qwen3.8:27b"), (64, "qwen3.8:27b"),
])
def test_local_llm_fits_memory(ram, llm):
    assert recommend_models(Hardware(ram, True)).local_llm == llm


def test_apple_silicon_uses_parakeet():
    assert recommend_models(Hardware(8, True)).asr_engine == "parakeet"


def test_intel_uses_whisper_sized_to_memory():
    assert recommend_models(Hardware(8, False)).asr_engine == "faster-whisper"
    assert recommend_models(Hardware(8, False)).whisper_model == "small"
    assert recommend_models(Hardware(16, False)).whisper_model == "large-v3-turbo"


def test_low_memory_summary_points_to_cloud():
    assert "cloud" in recommend_models(Hardware(8, True)).summary
