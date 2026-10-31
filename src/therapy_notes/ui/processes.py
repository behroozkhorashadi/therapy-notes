"""Helpers for starting the app's own Python processes from the UI."""
from __future__ import annotations

import sys
from pathlib import Path

from PyQt6.QtCore import QProcess, QProcessEnvironment

SRC_DIR = Path(__file__).resolve().parents[2]


def make_process(parent, module: str, *args: str) -> QProcess:
    """A QProcess, not yet started, that runs `python -m therapy_notes.<module>`."""
    process = QProcess(parent)
    env = QProcessEnvironment.systemEnvironment()
    existing = env.value("PYTHONPATH", "")
    env.insert("PYTHONPATH", f"{SRC_DIR}:{existing}" if existing else str(SRC_DIR))
    process.setProcessEnvironment(env)
    process.setWorkingDirectory(str(SRC_DIR.parent))
    process.setProgram(sys.executable)
    process.setArguments(["-m", f"therapy_notes.{module}", *args])
    return process


def module_command(module: str, *args: str) -> list[str]:
    return [sys.executable, "-m", f"therapy_notes.{module}", *args]


def module_env() -> dict:
    import os
    env = dict(os.environ)
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{SRC_DIR}:{existing}" if existing else str(SRC_DIR)
    return env
