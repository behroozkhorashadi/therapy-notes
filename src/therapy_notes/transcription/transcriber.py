"""Local speech-to-text. No audio leaves the machine in this step."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from .engines import Engine, Tick, create_engine


class Transcriber:
    def __init__(
        self,
        model_size: str = "small",
        engine: str = "faster-whisper",
        tick: Tick = None,
    ) -> None:
        self.model_size = model_size
        self.engine_name = engine
        self.tick = tick
        self._engine: Optional[Engine] = None

    def _load(self) -> Engine:
        if self._engine is None:
            self._engine = create_engine(self.engine_name, self.model_size)
        return self._engine

    def _get_timed_segments(self, audio_path: Path) -> list[tuple[float, str]]:
        """Return [(start_seconds, text), ...] for all speech in the file."""
        return self._load().segments(audio_path, self.tick)

    def transcribe(self, audio_path: Path) -> str:
        return " ".join(text for _, text in self._get_timed_segments(audio_path))

    def transcribe_two_speakers(self, therapist_path: Path, client_path: Path) -> str:
        """Transcribe both tracks and interleave them into a labelled transcript:

            Therapist: How are you feeling today?

            Client: I've been really anxious all week...
        """
        therapist = self._get_timed_segments(therapist_path)
        client = self._get_timed_segments(client_path)
        tagged = [(start, "Therapist", text) for start, text in therapist]
        tagged += [(start, "Client", text) for start, text in client]
        tagged.sort(key=lambda item: item[0])
        if not tagged:
            return ""

        lines: list[str] = []
        speaker = tagged[0][1]
        parts = [tagged[0][2]]
        for _, next_speaker, text in tagged[1:]:
            if next_speaker == speaker:
                parts.append(text)
            else:
                lines.append(f"{speaker}: {' '.join(parts)}")
                speaker, parts = next_speaker, [text]
        lines.append(f"{speaker}: {' '.join(parts)}")
        return "\n\n".join(lines)
