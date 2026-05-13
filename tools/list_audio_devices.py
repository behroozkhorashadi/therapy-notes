#!/usr/bin/env python3
"""
Run this to see every audio device on the machine:

    python tools/list_audio_devices.py

Use the name or index in your .env file:
    AUDIO_PRIMARY_DEVICE=   ← leave blank for system default mic
    AUDIO_LOOPBACK_DEVICE=BlackHole 2ch
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import sounddevice as sd


def main() -> None:
    devices = sd.query_devices()
    print("\nAll audio devices")
    print("=" * 64)
    for i, dev in enumerate(devices):
        tags = []
        if dev["max_input_channels"] > 0:
            tags.append(f"IN({dev['max_input_channels']}ch)")
        if dev["max_output_channels"] > 0:
            tags.append(f"out({dev['max_output_channels']}ch)")
        print(f"  [{i:2d}]  {dev['name']}")
        print(f"         {' | '.join(tags)}")

    print()
    try:
        d = sd.query_devices(kind="input")
        print(f"Default input  : {d['name']}")
    except Exception:
        pass
    try:
        d = sd.query_devices(kind="output")
        print(f"Default output : {d['name']}")
    except Exception:
        pass
    print()


if __name__ == "__main__":
    main()
