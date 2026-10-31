from __future__ import annotations

"""
Sends the transcript + SOAP template to the chosen AI provider and returns
completed session notes. Cloud providers receive the PII-scrubbed transcript;
Ollama runs on this machine and receives the raw one.

Any incomplete result raises NoteGenerationError. A cut-off or refused note must
never be saved as if it were complete.
"""

import json
import logging
import urllib.error
import urllib.request

import anthropic
import openai

from ..config import config


log = logging.getLogger(__name__)


class NoteGenerationError(RuntimeError):
    pass


# Enough for a long session's note, with streaming so long replies can't time out.
_MAX_TOKENS = 16_000

# These models accept effort and server-side refusal fallback.
_FALLBACK_MODELS = {"claude-fable-5-1", "claude-opus-5-5", "claude-opus-5", "claude-sonnet-5-5"}

_RISK_INDIVIDUAL = """\
- Risk assessment fields (suicidal/homicidal ideation, risk level) may only be \
  completed from what was actually said. Mark "Denied" only if the client \
  explicitly denied it in the transcript. If it was not discussed, leave the \
  boxes unchecked and write "Not assessed in this recording — clinician to \
  complete." Never infer a denial or a risk level from silence.
"""

_RISK_COUPLES = """\
- Risk assessment fields must be handled separately for each partner, and only \
  from what was actually said. Mark "Denied" only if that partner explicitly \
  denied it in the transcript. If it was not discussed, leave the boxes \
  unchecked and write "Not assessed in this recording — clinician to complete." \
  Never infer a denial or a risk level from silence.
"""

_PII_ANONYMIZED = (
    "You will receive an anonymized transcript of a telehealth {session_type} session. "
    "PII has been replaced with placeholders like [PERSON], [DATE], [LOCATION] — "
    "treat these as stand-ins for the real values when completing the note."
)

_PII_NOT_ANONYMIZED = (
    "You will receive a transcript of a telehealth {session_type} session. "
    "The transcript contains real patient information — handle it with strict confidentiality."
)

_INDIVIDUAL_BODY = """\

The transcript is speaker-labeled. Each turn begins with "Therapist:" or \
"Client:" followed by what that person said. Use these labels to correctly \
attribute content: the Subjective section should draw from Client turns, \
and the Objective section should reflect the therapist's observations from \
Therapist turns.

Your task is to fill in the provided SOAP note template based solely on what \
appears in the transcript. Follow these rules:

- Be professional, concise, and clinically precise.
- Do NOT infer or fabricate information not present in the transcript.
- Use "[not addressed this session]" for sections with no relevant content.
- Use "[unclear from recording]" when the transcript is garbled or ambiguous.
- Preserve the template's structure and formatting exactly — only fill in \
  the blank areas.
""" + _RISK_INDIVIDUAL

_COUPLES_BODY = """\

The transcript is speaker-labeled. Each turn begins with "Therapist:" or \
"Client:" followed by what that person said. Because both partners share the \
same audio channel, both appear under "Client:" — use context clues from the \
therapist's language (e.g., addressing partners by name or as "you" vs. "you \
two", turn-taking patterns, direct quotes) to distinguish Partner 1 from \
Partner 2 wherever possible. When you cannot reliably attribute a statement to \
a specific partner, write it as a shared or unattributed client observation.

Your task is to fill in the provided couples SOAP note template based solely \
on what appears in the transcript. Follow these rules:

- Be professional, concise, and clinically precise.
- Focus on relational dynamics, communication patterns, and dyadic themes in \
  addition to each partner's individual presentation.
- Do NOT infer or fabricate information not present in the transcript.
- Use "[not addressed this session]" for sections with no relevant content.
- Use "[unclear from recording]" when the transcript is garbled or ambiguous.
- Preserve the template's structure and formatting exactly — only fill in \
  the blank areas.
""" + _RISK_COUPLES


def _build_system_prompt(couples: bool, anonymized: bool) -> str:
    session_type = "couples therapy" if couples else "therapy"
    pii_line = _PII_ANONYMIZED if anonymized else _PII_NOT_ANONYMIZED
    header = (
        "You are an expert clinical documentation assistant helping a Licensed Clinical "
        f"Social Worker (LCSW) write {'couples ' if couples else ''}therapy session notes.\n\n"
        + pii_line.format(session_type=session_type)
    )
    body = _COUPLES_BODY if couples else _INDIVIDUAL_BODY
    return header + body


# Default system prompt (individual session, anonymized transcript).
# Exposed as a module constant so tests can make assertions against it.
_SYSTEM_PROMPT = _build_system_prompt(couples=False, anonymized=True)


class NoteGenerator:
    def __init__(self, couples: bool = False, anonymized: bool = True) -> None:
        template_path = config.couples_template_file if couples else config.template_file
        if not template_path.exists():
            raise FileNotFoundError(f"Template not found: {template_path}")
        self._template = template_path.read_text(encoding="utf-8")
        self._anonymized = anonymized
        self._system_prompt = _build_system_prompt(couples=couples, anonymized=anonymized)

    def generate(self, transcript: str) -> str:
        label = "Anonymized session transcript" if self._anonymized else "Session transcript"
        prompt = (
            f"{label}:\n\n"
            f"<transcript>\n{transcript}\n</transcript>\n\n"
            "Please complete the following SOAP note template based on this session:\n\n"
            f"<template>\n{self._template}\n</template>"
        )
        if config.ai_provider == "anthropic":
            return self._call_anthropic(prompt)
        if config.ai_provider == "ollama":
            return self._call_ollama(prompt)
        return self._call_openai(prompt)

    def _call_anthropic(self, prompt: str) -> str:
        client = anthropic.Anthropic(api_key=config.anthropic_api_key)
        model = config.ai_model_anthropic
        request: dict = {
            "model": model,
            "max_tokens": _MAX_TOKENS,
            "system": self._system_prompt,
            "messages": [{"role": "user", "content": prompt}],
        }
        message = None
        if model in _FALLBACK_MODELS:
            # Clinical content can trip safety classifiers; fall back, not fail.
            try:
                with client.beta.messages.stream(
                    **request,
                    output_config={"effort": "high"},
                    betas=["server-side-fallback-2026-07-01"],
                    extra_body={"fallbacks": "default"},
                ) as stream:
                    message = stream.get_final_message()
            except anthropic.BadRequestError as exc:
                # An optional feature being rejected must not stop the note.
                log.warning("Fallback request rejected (%s); retrying without it", exc.status_code)
        if message is None:
            with client.messages.stream(**request) as stream:
                message = stream.get_final_message()
        if message.stop_reason == "refusal":
            raise NoteGenerationError("The AI provider declined to write this note. Write it manually or retry.")
        if message.stop_reason == "max_tokens":
            raise NoteGenerationError("The note was cut off before it finished. Retry the session.")
        text = "".join(block.text for block in message.content if block.type == "text").strip()
        if not text:
            raise NoteGenerationError("The AI provider returned an empty note.")
        return text

    def _call_openai(self, prompt: str) -> str:
        client = openai.OpenAI(api_key=config.openai_api_key)
        response = client.chat.completions.create(
            model=config.ai_model_openai,
            messages=[
                {"role": "system", "content": self._system_prompt},
                {"role": "user", "content": prompt},
            ],
            max_completion_tokens=_MAX_TOKENS,
        )
        choice = response.choices[0]
        if choice.finish_reason == "length":
            raise NoteGenerationError("The note was cut off before it finished. Retry the session.")
        text = (choice.message.content or "").strip()
        if not text:
            raise NoteGenerationError("The AI provider returned an empty note.")
        return text

    def _call_ollama(self, prompt: str) -> str:
        """Use Ollama's native API, the way to set the context window per request.

        Ollama's default window is smaller than a long session's transcript, and it
        silently drops what doesn't fit, so size the window to the prompt and verify.
        """
        total_chars = len(self._system_prompt) + len(prompt)
        estimated_tokens = total_chars // 3
        body = {
            "model": config.ollama_model,
            "stream": False,
            "keep_alive": "2m",
            "messages": [
                {"role": "system", "content": self._system_prompt},
                {"role": "user", "content": prompt},
            ],
            "options": {
                "num_ctx": min(max(8192, estimated_tokens + 6000), 65536),
                "num_predict": 6000,
            },
        }
        request = urllib.request.Request(
            f"{config.ollama_base_url}/api/chat",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=1800) as response:
                reply = json.load(response)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:200]
            raise NoteGenerationError(f"Ollama error: {detail}") from exc
        except OSError as exc:
            raise NoteGenerationError(f"Could not reach Ollama at {config.ollama_base_url}: {exc}") from exc

        if reply.get("done_reason") == "length":
            raise NoteGenerationError("The note was cut off before it finished. Retry the session.")
        evaluated = reply.get("prompt_eval_count") or 0
        # Text averages about 4 characters per token; well under that means truncation.
        if evaluated and evaluated < total_chars / 8:
            raise NoteGenerationError("The transcript didn't fit in the local model's memory, so no note was saved.")
        text = (reply.get("message", {}).get("content") or "").strip()
        if not text:
            raise NoteGenerationError("The local model returned an empty note.")
        return text
