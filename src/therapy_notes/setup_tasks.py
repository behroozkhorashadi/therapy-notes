"""Audio tests for the Setup window, each run as its own process: 
python -m therapy_notes.setup_tasks devices|mic-test|loopback-test

Keeps audio drivers and speech models out of the UI process. Prints one JSON object.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import sys
import tempfile
import time
from pathlib import Path

MIC_SCRIPT = (
    "Good morning. Testing the audio recording system. "
    "My name is Sarah and I am conducting a telehealth session today. "
    "One, two, three, four, five."
)
MIC_RECORD_SECONDS = 10


def word_match(expected: str, actual: str) -> float:
    """Return fraction of expected words found anywhere in actual."""
    def words(text: str) -> set[str]:
        return set(re.sub(r"[^\w\s]", "", text.lower()).split())

    exp = words(expected)
    return len(exp & words(actual)) / len(exp) if exp else 0.0


def devices() -> dict:
    from .audio.devices import find_device
    return {"blackhole": find_device("BlackHole 2ch", kind="input") is not None}


def mic_test() -> dict:
    from .audio.recorder import StreamingRecorder
    from .config import config
    from .transcription.transcriber import Transcriber

    folder = Path(tempfile.mkdtemp())
    try:
        recorder = StreamingRecorder(folder, config.audio_primary_device)
        recorder.start()
        time.sleep(MIC_RECORD_SECONDS)
        recorder.stop()
        text = Transcriber(config.whisper_model, config.asr_engine).transcribe(recorder.tracks[0].path)
        return {"transcript": text.strip(), "ratio": word_match(MIC_SCRIPT, text)}
    finally:
        shutil.rmtree(folder, ignore_errors=True)


def loopback_test() -> dict:
    """Play a tone to 'Therapy Output' and check that BlackHole 2ch captures it."""
    import numpy as np
    import sounddevice as sd
    from .audio.devices import find_device

    rate, seconds = 16_000, 3
    blackhole = find_device("BlackHole 2ch", kind="input")
    output = find_device("Therapy Output", kind="output")
    if blackhole is None:
        return {"error": "BlackHole 2ch input device not found. Complete Step 1 and restart Core Audio."}
    if output is None:
        return {"error": "'Therapy Output' device not found. Create the Multi-Output Device in Step 2 first."}

    t = np.linspace(0, seconds, rate * seconds, endpoint=False)
    tone = (0.4 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    recorded: list = []
    with sd.InputStream(samplerate=rate, channels=1, device=blackhole, dtype="float32",
                        callback=lambda data, *_: recorded.append(data.copy())):
        sd.play(tone, samplerate=rate, device=output, blocking=True)
        time.sleep(0.3)
    peak = float(np.max(np.abs(np.concatenate(recorded)))) if recorded else 0.0
    if peak >= 0.005:
        return {"transcript": f"Tone captured cleanly  (peak={peak:.3f})", "ratio": 1.0}
    return {
        "transcript": f"Signal too weak  (peak={peak:.4f})\nMake sure 'Therapy Output' is the speaker "
                      "output and BlackHole 2ch is checked in the Multi-Output Device.",
        "ratio": 0.0,
    }


def main() -> int:
    tasks = {"devices": devices, "mic-test": mic_test, "loopback-test": loopback_test}
    try:
        result = tasks[sys.argv[1]]()
    except Exception as exc:  # noqa: BLE001
        result = {"error": str(exc)}
    print("\n" + json.dumps(result))
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    raise SystemExit(main())
