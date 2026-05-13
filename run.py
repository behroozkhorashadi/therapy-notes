#!/usr/bin/env python3
"""
Entry point — run this to launch the app:

    uv run python run.py

First-time setup:
    1. uv sync
    2. uv run python run.py   # the setup wizard handles everything else
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from therapy_notes.main import main

if __name__ == "__main__":
    main()
