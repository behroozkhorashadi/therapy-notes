"""Turns a recorded session into notes, one resumable stage at a time.

Each stage saves its result before the next begins, so a restart continues where
it stopped. Audio is deleted only after notes are safely saved.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Optional

from . import session as store
from .config import config

log = logging.getLogger(__name__)

Wait = Callable[[], None]


class PipelineError(RuntimeError):
    pass


class LocalBackend:
    """Runs speech-to-text on this Mac.

    A remote backend (a Mac mini, for example) would implement the same `transcribe`
    method by uploading the audio and waiting for the transcript.
    """

    def __init__(self, tick: Optional[Wait] = None) -> None:
        self._tick = tick
        self._transcriber = None
        self._anonymizer = None

    def transcribe(self, therapist: Path, client: Optional[Path]) -> tuple[str, bool]:
        """Return the transcript and whether the client track contained speech."""
        from .transcription.transcriber import Transcriber

        if self._transcriber is None:
            self._transcriber = Transcriber(config.whisper_model, config.asr_engine, self._tick)
        if client is None:
            return self._transcriber.transcribe(therapist), False
        text = self._transcriber.transcribe_two_speakers(therapist, client)
        return text, "Client:" in text

    def anonymize(self, transcript: str) -> str:
        from .anonymization.anonymizer import Anonymizer

        if self._anonymizer is None:
            self._anonymizer = Anonymizer()
        return self._anonymizer.anonymize(transcript)

    def generate_notes(self, transcript: str, couples: bool, anonymized: bool) -> str:
        from .notes.generator import NoteGenerator

        return NoteGenerator(couples=couples, anonymized=anonymized).generate(transcript)


def process_session(session_dir: Path, backend, wait: Wait = lambda: None) -> None:
    """Run every unfinished stage. Raises on failure, leaving earlier results saved."""
    data = store.load_session(session_dir)
    raw_path = session_dir / store.RAW_TRANSCRIPT
    anon_path = session_dir / store.ANON_TRANSCRIPT
    notes_path = session_dir / store.NOTES_FILE
    local_only = config.ai_provider == "ollama"

    if not (notes_path.exists() or raw_path.exists() or anon_path.exists()):
        wait()
        store.update_session(session_dir, state=store.TRANSCRIBING)
        therapist = session_dir / store.THERAPIST_AUDIO
        client = session_dir / store.CLIENT_AUDIO
        if not therapist.exists():
            raise PipelineError("This session's audio is missing.")
        transcript, client_speech = backend.transcribe(therapist, client if client.exists() else None)
        if not transcript.strip():
            raise PipelineError("No speech was found in the recording, so no note was written.")
        if not client.exists():
            store.add_warning(session_dir, "Only the microphone was recorded. Check this note carefully: "
                                           "the client's words may be missing and speakers aren't separated.")
        elif not client_speech:
            store.add_warning(session_dir, "No client speech was found. Check Zoom's speaker setting: "
                                           "this note may reflect only the therapist's side.")
        store.atomic_write_text(raw_path, transcript)

    if not notes_path.exists():
        if not local_only and not anon_path.exists():
            wait()
            store.update_session(session_dir, state=store.ANONYMIZING)
            store.atomic_write_text(anon_path, backend.anonymize(raw_path.read_text(encoding="utf-8")))

        wait()
        store.update_session(session_dir, state=store.GENERATING)
        source = raw_path if local_only else anon_path
        if not source.exists():
            raise PipelineError("The transcript this provider needs is missing. Switch AI provider in Setup and retry.")
        couples = bool(store.load_session(session_dir).get("couples"))
        notes = backend.generate_notes(source.read_text(encoding="utf-8"), couples, not local_only)
        store.atomic_write_text(notes_path, notes)

    if config.delete_raw_transcript and anon_path.exists():
        raw_path.unlink(missing_ok=True)
    if config.delete_audio_after_notes:
        store.delete_audio(session_dir)
    store.update_session(session_dir, state=store.DONE, error=None, failed_stage=None)
    log.info("Session %s finished", data["id"])
