from __future__ import annotations

"""
Local speech-to-text using OpenAI Whisper via faster-whisper.
No audio data ever leaves the machine.

Model size guide (Apple Silicon, CPU + int8):
  tiny   — ~30s for 50min audio, noticeably less accurate
  base   — ~1–2min, decent accuracy
  small  — ~3–5min, good accuracy  ← default
  medium — ~8–12min, great accuracy (recommended if accuracy matters most)

The model is downloaded once (~150MB for small) and cached in
~/.cache/huggingface/hub/
"""

from pathlib import Path
from typing import Callable, Optional

from faster_whisper import WhisperModel

# Whisper uses this string as context before the audio begins, biasing its
# vocabulary toward these terms without changing the model or requiring
# fine-tuning. Written as natural prose so Whisper treats it as a prior
# transcript snippet rather than a raw word list.
_CLINICAL_PROMPT = (
    "Telehealth therapy session. The client presents with generalized anxiety "
    "disorder, major depressive disorder, PTSD, trauma, OCD, ADHD, bipolar "
    "disorder, borderline personality disorder, dysthymia, or adjustment disorder. "
    "Modalities discussed include CBT, DBT, EMDR, ACT, psychodynamic therapy, "
    "mindfulness, somatic work, and motivational interviewing. "
    "Medications mentioned: SSRIs, SNRIs, Lexapro, escitalopram, Zoloft, sertraline, "
    "Prozac, fluoxetine, Effexor, venlafaxine, Wellbutrin, bupropion, Cymbalta, "
    "duloxetine, Lamictal, lamotrigine, Seroquel, quetiapine, Abilify, aripiprazole, "
    "Klonopin, clonazepam, Xanax, alprazolam, Ativan, lorazepam, Adderall, Vyvanse. "
    "Clinical terms: affect, mood, insight, judgment, suicidal ideation, SI, "
    "homicidal ideation, HI, self-harm, dissociation, hypervigilance, rumination, "
    "alexithymia, attachment, avoidance, cognitive distortions, coping skills, "
    "psychoeducation, safety plan, treatment plan, therapeutic alliance, "
    "transference, countertransference, containment, grounding, window of tolerance, "
    "polyvagal, nervous system, regulation, dysregulation, boundaries, triggers, "
    "flashbacks, intrusive thoughts, compulsions, obsessions, panic attack."
)


class Transcriber:
    def __init__(self, model_size: str = "small") -> None:
        self.model_size = model_size
        self._model: Optional[WhisperModel] = None

    def _load(self) -> None:
        if self._model is None:
            # faster-whisper does not support MPS (Apple Silicon GPU).
            # CPU + int8 is the correct choice on macOS.
            self._model = WhisperModel(
                self.model_size,
                device="cpu",
                compute_type="int8",
            )

    def transcribe(
        self,
        audio_path: Path,
        on_progress: Optional[Callable[[float], None]] = None,
    ) -> str:
        """
        Transcribe audio_path and return the full transcript as a string.

        on_progress: optional callback(fraction: 0.0–1.0) called after each segment.
        """
        self._load()

        segments, info = self._model.transcribe(
            str(audio_path),
            language="en",
            beam_size=5,
            vad_filter=True,                          # skip long silences
            vad_parameters={"min_silence_duration_ms": 500},
            initial_prompt=_CLINICAL_PROMPT,
        )

        parts: list[str] = []
        for seg in segments:
            parts.append(seg.text.strip())
            if on_progress and info.duration:
                on_progress(min(seg.end / info.duration, 1.0))

        return " ".join(parts)

    def transcribe_two_speakers(
        self,
        therapist_path: Path,
        client_path: Path,
    ) -> str:
        """
        Transcribe two audio files recorded simultaneously and interleave
        the segments by timestamp to produce a speaker-labelled transcript:

            Therapist: How are you feeling today?

            Client: I've been really anxious all week...
        """
        therapist_segs = self._get_timed_segments(therapist_path)
        client_segs = self._get_timed_segments(client_path)

        # Tag and sort all segments by start time
        tagged: list[tuple[float, str, str]] = (
            [(start, "Therapist", text) for start, text in therapist_segs] +
            [(start, "Client",    text) for start, text in client_segs]
        )
        tagged.sort(key=lambda x: x[0])

        if not tagged:
            return ""

        # Merge consecutive runs from the same speaker into one paragraph
        lines: list[str] = []
        cur_speaker = tagged[0][1]
        cur_parts: list[str] = [tagged[0][2]]

        for _, speaker, text in tagged[1:]:
            if speaker == cur_speaker:
                cur_parts.append(text)
            else:
                lines.append(f"{cur_speaker}: {' '.join(cur_parts)}")
                cur_speaker = speaker
                cur_parts = [text]

        lines.append(f"{cur_speaker}: {' '.join(cur_parts)}")
        return "\n\n".join(lines)

    def _get_timed_segments(self, audio_path: Path) -> list[tuple[float, str]]:
        """Return [(start_seconds, text), ...] for all non-silent segments."""
        self._load()
        segments, _ = self._model.transcribe(
            str(audio_path),
            language="en",
            beam_size=5,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 500},
            initial_prompt=_CLINICAL_PROMPT,
        )
        return [(seg.start, seg.text.strip()) for seg in segments if seg.text.strip()]
