from __future__ import annotations

"""
Sends the anonymized transcript + SOAP template to an AI API and returns
completed session notes. This is the only step that touches the internet.
"""

import anthropic
import openai

from ..config import config

# Ollama exposes an OpenAI-compatible endpoint — no extra dependency needed.

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
- Risk assessment fields (suicidal/homicidal ideation) must always be \
  completed. If not mentioned in the session, write \
  "Denied — not reported in session."
"""

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
- Risk assessment fields must be completed for each partner. If not mentioned \
  in the session, write "Denied — not reported in session."
"""


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
        self._template = template_path.read_text()
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
        message = client.messages.create(
            model=config.ai_model_anthropic,
            max_tokens=2048,
            system=self._system_prompt,
            messages=[{"role": "user", "content": prompt}],
        )
        return message.content[0].text

    def _call_openai_compatible(
        self,
        prompt: str,
        *,
        api_key: str,
        model: str,
        base_url: str | None = None,
    ) -> str:
        """Shared implementation for OpenAI-protocol providers (OpenAI and Ollama)."""
        client = openai.OpenAI(api_key=api_key, base_url=base_url)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": self._system_prompt},
                {"role": "user", "content": prompt},
            ],
            max_tokens=2048,
        )
        return response.choices[0].message.content

    def _call_openai(self, prompt: str) -> str:
        return self._call_openai_compatible(
            prompt,
            api_key=config.openai_api_key,
            model=config.ai_model_openai,
        )

    def _call_ollama(self, prompt: str) -> str:
        return self._call_openai_compatible(
            prompt,
            api_key="ollama",  # required by SDK, not used by Ollama
            model=config.ollama_model,
            base_url=f"{config.ollama_base_url}/v1",
        )
