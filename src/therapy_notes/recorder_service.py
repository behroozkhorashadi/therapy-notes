"""Recorder process, run by the UI: python -m therapy_notes.recorder_service

Commands arrive on stdin ("stop") and JSON events leave on stdout. If the UI goes
away, stdin closes and the recording is saved rather than lost. Running apart from
the UI and the transcription worker keeps audio capture from being starved.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

from .audio.recorder import StreamingRecorder
from .config import config
from .logging_setup import setup_logging
from .session import (
    QUEUED,
    add_warning,
    clear_marker,
    create_session,
    update_session,
    write_marker,
)

log = logging.getLogger("recorder")
_print_lock = threading.Lock()
MIN_SECONDS = 1.0


def _send(event: str, **fields) -> None:
    with _print_lock:
        try:
            sys.stdout.write(json.dumps({"event": event, **fields}) + "\n")
            sys.stdout.flush()
        except (BrokenPipeError, ValueError, OSError):
            pass    # the UI is gone; keep going so the audio still gets saved


def _resolve(name):
    """Turn a saved device name into a device index, or None if it's missing."""
    from .audio.devices import find_device
    return find_device(name, kind="input")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--couples", action="store_true")
    parser.add_argument("--max-seconds", type=float, default=4 * 3600)
    args = parser.parse_args()

    setup_logging("recorder")
    root: Path = config.sessions_dir
    session_dir = create_session(root, couples=args.couples)
    write_marker(root, session_dir.name)
    log.info("Recording session %s", session_dir.name)

    if shutil.which("caffeinate"):
        subprocess.Popen(["caffeinate", "-i", "-w", str(os.getpid())])

    done = threading.Event()
    stop_reason: dict = {"code": None}

    def on_event(kind: str, code: str, message: str) -> None:
        if kind == "status":
            _send("status", **recorder.levels())
        elif kind == "warning":
            log.warning("Recording warning: %s", code)
            add_warning(session_dir, message)
            _send("warning", code=code, message=message)
        elif kind == "fatal":
            stop_reason["code"] = code
            add_warning(session_dir, message)
            _send("warning", code=code, message=message)
            done.set()

    try:
        primary_name = config.audio_primary_device
        primary = _resolve(primary_name) if primary_name else None
        if primary_name and primary is None:
            message = f"Microphone '{primary_name}' wasn't found, so the system default is recording."
            add_warning(session_dir, message)
            _send("warning", code="mic_default", message=message)
        loopback = config.audio_loopback_device
        loopback_index = _resolve(loopback) if loopback else None
        if loopback and loopback_index is None:
            message = f"'{loopback}' wasn't found, so client audio isn't being recorded. Check Setup."
            add_warning(session_dir, message)
            _send("warning", code="client_unavailable", message=message)
        recorder = StreamingRecorder(session_dir, primary, loopback_index, emit=on_event)
        recorder.start()
    except Exception as exc:  # noqa: BLE001
        log.exception("Could not start recording")
        clear_marker(root)
        shutil.rmtree(session_dir, ignore_errors=True)
        _send("error", message=f"Could not start recording: {exc}")
        return 1

    _send("started", session=session_dir.name)

    def read_stdin() -> None:
        for line in sys.stdin:
            if line.strip() == "stop":
                break
        done.set()

    threading.Thread(target=read_stdin, daemon=True).start()
    started = time.monotonic()
    while not done.wait(1.0):
        if time.monotonic() - started >= args.max_seconds:
            message = "Recording stopped at the safety limit. The audio is saved."
            add_warning(session_dir, message)
            _send("warning", code="max_length", message=message)
            break

    summary = recorder.stop()
    if summary["seconds"] < MIN_SECONDS:
        log.info("Discarding empty recording %s", session_dir.name)
        clear_marker(root)
        shutil.rmtree(session_dir, ignore_errors=True)
        _send("stopped", discarded=True)
    else:
        for warning in summary["warnings"]:
            add_warning(session_dir, warning)
        update_session(session_dir, state=QUEUED, seconds=summary["seconds"])
        clear_marker(root)
        log.info("Saved session %s (%.0f s)", session_dir.name, summary["seconds"])
        _send("stopped", session=session_dir.name, seconds=summary["seconds"])
    sys.stdout.flush()
    os._exit(0)   # don't wait on an audio thread that may be hung


if __name__ == "__main__":
    raise SystemExit(main())
