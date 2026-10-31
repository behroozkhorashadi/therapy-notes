"""Log files and crash capture, shared by the UI, recorder and worker.

Logs hold session ids, stages and errors. Never log transcript, note or
patient content.
"""
from __future__ import annotations

import faulthandler
import logging
import logging.handlers
import sys
from pathlib import Path

from .config import data_dir

_fault_log = None


def log_path() -> Path:
    return data_dir() / "logs" / "therapy-notes.log"


def setup_logging(process_name: str) -> None:
    global _fault_log
    path = log_path()
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    handler = logging.handlers.RotatingFileHandler(path, maxBytes=2_000_000, backupCount=3, encoding="utf-8")
    handler.setFormatter(logging.Formatter(f"%(asctime)s {process_name} %(levelname)s %(name)s: %(message)s"))
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.addHandler(handler)
    for noisy in ("httpx", "httpx2", "presidio-analyzer", "faster_whisper"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _fault_log = open(path.parent / "crashes.log", "a", encoding="utf-8")
    faulthandler.enable(file=_fault_log)

    def _hook(exc_type, exc, tb):
        logging.getLogger(process_name).critical("Unhandled exception", exc_info=(exc_type, exc, tb))

    sys.excepthook = _hook
