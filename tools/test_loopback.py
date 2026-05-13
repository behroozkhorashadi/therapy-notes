#!/usr/bin/env python3
"""
Test the BlackHole loopback: plays a tone to 'Therapy Output' (Multi-Output Device)
and records from 'BlackHole 2ch' simultaneously. If audio is captured, the test passes.

Usage:
    uv run python tools/test_loopback.py
"""
import sys
import time
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import sounddevice as sd
import scipy.io.wavfile as wavfile

from therapy_notes.audio.devices import find_device

SAMPLE_RATE = 16_000
RECORD_SECONDS = 4
TONE_HZ = 440  # A4


def main():
    print("=== BlackHole Loopback Test ===\n")

    # Find devices
    blackhole_idx = find_device("BlackHole 2ch", kind="input")
    therapy_idx = find_device("Therapy Output", kind="output")

    if blackhole_idx is None:
        print("ERROR: 'BlackHole 2ch' input device not found.")
        print("Make sure BlackHole is installed and Core Audio has loaded it.")
        sys.exit(1)

    if therapy_idx is None:
        print("ERROR: 'Therapy Output' (Multi-Output Device) not found.")
        print("Create the Multi-Output Device in Audio MIDI Setup (Step 2 in Setup & Test).")
        sys.exit(1)

    print(f"  BlackHole 2ch  → device [{blackhole_idx}] (recording from)")
    print(f"  Therapy Output → device [{therapy_idx}] (playing to)")
    print()

    # Generate a 440 Hz tone
    t = np.linspace(0, RECORD_SECONDS, int(SAMPLE_RATE * RECORD_SECONDS), endpoint=False)
    tone = (0.4 * np.sin(2 * np.pi * TONE_HZ * t)).astype(np.float32)

    # Record captured audio
    recorded = []

    def record_callback(indata, frames, time_info, status):
        if status:
            print(f"  [rec] {status}")
        recorded.append(indata.copy())

    print(f"Playing 440 Hz tone to Therapy Output and recording from BlackHole for {RECORD_SECONDS}s...")

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        device=blackhole_idx,
        callback=record_callback,
        dtype="float32",
    ):
        sd.play(tone, samplerate=SAMPLE_RATE, device=therapy_idx, blocking=True)
        # Give the stream a moment to flush
        time.sleep(0.3)

    if not recorded:
        print("\nERROR: Nothing was recorded from BlackHole 2ch.")
        sys.exit(1)

    captured = np.concatenate(recorded, axis=0).flatten()

    # Check signal level
    rms = float(np.sqrt(np.mean(captured ** 2)))
    peak = float(np.max(np.abs(captured)))
    print(f"\nCaptured {len(captured)} samples  |  RMS={rms:.5f}  Peak={peak:.5f}")

    THRESHOLD = 0.005  # near-silence if below this
    if peak < THRESHOLD:
        print("\nRESULT: FAIL — audio captured but signal is near-silent.")
        print("Make sure 'Therapy Output' is the Multi-Output Device that includes BlackHole 2ch.")
        print("Also confirm that BlackHole 2ch is checked in the Multi-Output Device device list.")
        sys.exit(1)
    else:
        # Save a debug WAV for inspection
        out_path = Path(__file__).parent / "loopback_test.wav"
        wavfile.write(str(out_path), SAMPLE_RATE, captured)
        print(f"\nRESULT: PASS — signal detected! (saved to {out_path.name} for review)")
        print("BlackHole loopback is working correctly.")


if __name__ == "__main__":
    main()
