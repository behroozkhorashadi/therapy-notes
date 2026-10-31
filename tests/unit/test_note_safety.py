"""Notes must never overstate: no invented denials, no silently cut-off results."""
from __future__ import annotations

import io
import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from therapy_notes.notes import generator as g
from therapy_notes.notes.generator import NoteGenerationError, NoteGenerator, _build_system_prompt


@pytest.mark.parametrize("couples", [False, True])
@pytest.mark.parametrize("anonymized", [False, True])
def test_prompt_never_charts_denial_by_default(couples, anonymized):
    prompt = " ".join(_build_system_prompt(couples=couples, anonymized=anonymized).split())
    assert "Denied — not reported" not in prompt
    assert "Not assessed in this recording" in prompt
    assert "explicitly denied" in prompt


@pytest.fixture
def gen(mock_config, tmp_path):
    cfg = mock_config(AI_MODEL_ANTHROPIC="claude-opus-5-5")
    with patch.object(g, "config", cfg):
        yield NoteGenerator(), cfg


def anthropic_reply(stop_reason, text="note"):
    message = SimpleNamespace(stop_reason=stop_reason, content=[SimpleNamespace(type="text", text=text)])
    stream = MagicMock()
    stream.__enter__.return_value.get_final_message.return_value = message
    return stream


def test_anthropic_success_and_request_shape(gen):
    generator, _ = gen
    with patch.object(g.anthropic, "Anthropic") as client:
        stream_fn = client.return_value.beta.messages.stream
        stream_fn.return_value = anthropic_reply("end_turn", "the note")
        assert generator._call_anthropic("prompt") == "the note"
    kwargs = stream_fn.call_args.kwargs
    assert kwargs["max_tokens"] == g._MAX_TOKENS
    assert kwargs["output_config"] == {"effort": "high"}
    assert kwargs["extra_body"] == {"fallbacks": "default"}
    assert "thinking" not in kwargs and "temperature" not in kwargs


def test_rejected_fallback_option_retries_plainly(gen):
    generator, _ = gen
    with patch.object(g.anthropic, "Anthropic") as client:
        rejected = g.anthropic.BadRequestError(
            "unsupported", response=MagicMock(status_code=400, headers={}), body=None,
        )
        client.return_value.beta.messages.stream.side_effect = rejected
        client.return_value.messages.stream.return_value = anthropic_reply("end_turn", "plain note")
        assert generator._call_anthropic("prompt") == "plain note"


def test_older_model_uses_plain_request(mock_config):
    cfg = mock_config(AI_MODEL_ANTHROPIC="claude-sonnet-4-6")
    with patch.object(g, "config", cfg):
        generator = NoteGenerator()
        with patch.object(g.anthropic, "Anthropic") as client:
            client.return_value.messages.stream.return_value = anthropic_reply("end_turn")
            generator._call_anthropic("prompt")
            assert "betas" not in client.return_value.messages.stream.call_args.kwargs


@pytest.mark.parametrize("reason,match", [("max_tokens", "cut off"), ("refusal", "declined")])
def test_anthropic_incomplete_raises(gen, reason, match):
    generator, _ = gen
    with patch.object(g.anthropic, "Anthropic") as client:
        client.return_value.beta.messages.stream.return_value = anthropic_reply(reason)
        with pytest.raises(NoteGenerationError, match=match):
            generator._call_anthropic("prompt")


def test_openai_truncation_raises(gen):
    generator, _ = gen
    choice = SimpleNamespace(finish_reason="length", message=SimpleNamespace(content="partial"))
    with patch.object(g.openai, "OpenAI") as client:
        client.return_value.chat.completions.create.return_value = SimpleNamespace(choices=[choice])
        with pytest.raises(NoteGenerationError, match="cut off"):
            generator._call_openai("prompt")


def ollama_reply(**fields):
    body = {"message": {"content": "note"}, "done_reason": "stop", "prompt_eval_count": 100_000, **fields}
    response = MagicMock()
    response.__enter__.return_value = io.BytesIO(json.dumps(body).encode())
    return response


def test_ollama_sizes_window_to_transcript(gen):
    generator, _ = gen
    prompt = "word " * 20_000
    with patch.object(g.urllib.request, "urlopen", return_value=ollama_reply()) as opened:
        assert generator._call_ollama(prompt) == "note"
    sent = json.loads(opened.call_args.args[0].data)
    assert sent["options"]["num_ctx"] >= len(prompt) // 3
    assert opened.call_args.args[0].full_url.endswith("/api/chat")


def test_ollama_silent_truncation_is_caught(gen):
    generator, _ = gen
    prompt = "word " * 20_000
    with patch.object(g.urllib.request, "urlopen", return_value=ollama_reply(prompt_eval_count=2048)):
        with pytest.raises(NoteGenerationError, match="didn't fit"):
            generator._call_ollama(prompt)


def test_ollama_output_cutoff_raises(gen):
    generator, _ = gen
    with patch.object(g.urllib.request, "urlopen", return_value=ollama_reply(done_reason="length")):
        with pytest.raises(NoteGenerationError, match="cut off"):
            generator._call_ollama("short")


def test_ollama_unreachable_is_clear(gen):
    generator, _ = gen
    with patch.object(g.urllib.request, "urlopen", side_effect=OSError("refused")):
        with pytest.raises(NoteGenerationError, match="Could not reach Ollama"):
            generator._call_ollama("short")
