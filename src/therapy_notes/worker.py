"""Background worker process: python -m therapy_notes.worker

Handles one session at a time at low priority, pauses while a recording is active,
and exits when the queue is empty so idle sessions hold no model memory. The UI
starts it when work appears. A lock file ensures only one runs.
"""
from __future__ import annotations

import fcntl
import logging
import os
import time
from pathlib import Path
from typing import Optional

from . import session as store
from .audio.recorder import repair_wav
from .config import config, data_dir, reload_config
from .logging_setup import setup_logging
from .pipeline import LocalBackend, process_session

log = logging.getLogger("worker")

MAX_ATTEMPTS = 3
_POLL_SECONDS = 2.0


class _AppClosed(Exception):
    """The UI went away. The job goes back in the queue for the next launch."""


def recover_interrupted_recordings(root: Path) -> None:
    """Rescue sessions whose recorder died before saving, so their audio gets processed."""
    active = store.active_recording(root)
    for session_dir, data in store.list_sessions(root):
        if data.get("state") != store.RECORDING or data["id"] == active:
            continue
        seconds = 0.0
        for path in store.audio_paths(session_dir):
            try:
                seconds = max(seconds, repair_wav(path))
            except (ValueError, OSError) as exc:
                log.warning("Could not repair audio in %s: %s", data["id"], exc)
        if seconds < 1.0:
            store.update_session(session_dir, state=store.FAILED, failed_stage="recording",
                                 error="The app stopped before any audio was saved.")
            continue
        log.warning("Recovered interrupted recording %s (%.0f s)", data["id"], seconds)
        store.add_warning(session_dir, "The recording ended unexpectedly. This note covers only the audio saved until then.")
        store.update_session(session_dir, state=store.QUEUED, seconds=seconds)


def next_job(root: Path) -> Optional[Path]:
    """The oldest session waiting to be processed."""
    for session_dir, data in reversed(store.list_sessions(root)):
        if data.get("state") in store.PENDING:
            return session_dir
    return None


def describe_error(exc: Exception) -> str:
    """A message she can act on. Provider messages stay out of the UI unless useful."""
    name = type(exc).__name__
    if name == "AuthenticationError":
        return "The API key was rejected. Check it in Setup, then retry."
    if name == "PermissionDeniedError":
        return "The API key doesn't have access to this model. Check Setup, then retry."
    if name == "NotFoundError":
        return "The AI model wasn't found. Check the model name in Setup, then retry."
    if name == "RateLimitError":
        return "The AI provider's limit was reached. Wait a while, then retry."
    if name in ("APIConnectionError", "APITimeoutError", "ConnectionError"):
        return "Couldn't reach the AI provider. Check the internet connection, then retry."
    if name == "InternalServerError":
        return "The AI provider had a problem. Retry in a few minutes."
    if isinstance(exc, MemoryError):
        return "This Mac ran out of memory. Close other apps, then retry."
    return str(exc)[:300] or name


class _Run:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.parent = os.getppid()
        self.backend = LocalBackend(tick=self.wait)

    def wait(self) -> None:
        """Called often during work: exit if the app closed, pause if it's recording."""
        paused = False
        while True:
            if os.getppid() != self.parent:
                raise _AppClosed
            if store.active_recording(self.root) is None:
                if paused:
                    log.info("Resuming")
                return
            if not paused:
                log.info("Pausing while a session is recorded")
                paused = True
            time.sleep(_POLL_SECONDS)

    def run_job(self, session_dir: Path) -> None:
        attempts = store.load_session(session_dir).get("attempts", 0) + 1
        if attempts > MAX_ATTEMPTS:
            # Earlier attempts killed the worker outright (out of memory, for example).
            log.error("Session %s interrupted %d times", session_dir.name, attempts - 1)
            store.update_session(session_dir, state=store.FAILED, failed_stage="processing",
                                 error="Processing kept getting interrupted. Close other apps, then retry.")
            return
        store.update_session(session_dir, attempts=attempts)
        log.info("Processing session %s (attempt %d)", session_dir.name, attempts)
        try:
            process_session(session_dir, self.backend, self.wait)
        except _AppClosed:
            store.update_session(session_dir, state=store.QUEUED, attempts=attempts - 1)
            raise
        except Exception as exc:  # noqa: BLE001 - one bad session must not stop the queue
            stage = store.load_session(session_dir).get("state")
            log.error("Session %s failed at %s: %s", session_dir.name, stage, type(exc).__name__, exc_info=True)
            store.update_session(session_dir, state=store.FAILED, failed_stage=stage, error=describe_error(exc))


def main() -> int:
    setup_logging("worker")
    lock_path = data_dir() / "worker.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock = open(lock_path, "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        return 0
    os.nice(10)

    root = config.sessions_dir
    run = _Run(root)
    try:
        while True:
            reload_config()     # picks up changes made in Setup
            recover_interrupted_recordings(root)
            job = next_job(root)
            if job is None:
                return 0
            run.wait()
            run.run_job(job)
    except _AppClosed:
        log.info("App closed; work will resume at next launch")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
