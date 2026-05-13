#!/usr/bin/env python3
"""
Generate fixture audio files for integration tests.

Uses macOS `say` to synthesize speech into 16 kHz mono WAV files.
Run once, commit the results:

    python tests/generate_fixtures.py

Requires macOS (uses the `say` command).
"""
import subprocess
import sys
from pathlib import Path

FIXTURES_DIR = Path(__file__).parent / "fixtures"
SAMPLE_RATE = 16000

THERAPIST_TEXT = "How are you feeling today? Tell me about your week."
CLIENT_TEXT = (
    "I've been feeling anxious. My name is Sarah Johnson "
    "and I live in Portland."
)


def generate(text: str, output: Path) -> None:
    """Use macOS `say` to generate a 16 kHz mono WAV."""
    # say outputs AIFF by default; use --data-format for raw PCM WAV
    subprocess.run(
        [
            "say",
            "-o", str(output),
            "--data-format=LEI16@16000",
            text,
        ],
        check=True,
    )
    print(f"  Created {output.name} ({output.stat().st_size:,} bytes)")


def main():
    FIXTURES_DIR.mkdir(exist_ok=True)

    print("Generating fixture audio files...\n")

    generate(THERAPIST_TEXT, FIXTURES_DIR / "therapist_sample.wav")
    generate(CLIENT_TEXT, FIXTURES_DIR / "client_sample.wav")

    print("\nDone. Commit these files to the repo.")


if __name__ == "__main__":
    main()
