"""On-disk session store. The session folder is the source of truth.

Each session folder holds session.json plus its audio, transcripts and notes.
Ownership keeps writers apart: the recorder owns "recording", the worker owns
the queued and processing states, and the UI only changes finished or failed
sessions. All writes are atomic, so a crash never leaves a half-written file.
"""
from __future__ import annotations

import fcntl
import json
import logging
import os
import tempfile
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Iterator, Optional

log = logging.getLogger(__name__)

RECORDING = "recording"
QUEUED = "queued"
TRANSCRIBING = "transcribing"
ANONYMIZING = "anonymizing"
GENERATING = "generating"
DONE = "done"
FAILED = "failed"
UNPROCESSED = "unprocessed"   # older session with audio but no notes; never auto-started

IN_PROGRESS = (TRANSCRIBING, ANONYMIZING, GENERATING)
PENDING = (QUEUED, *IN_PROGRESS)

STATE_FILE = "session.json"
NOTES_FILE = "notes.txt"
RAW_TRANSCRIPT = "transcript_raw.txt"
ANON_TRANSCRIPT = "transcript_anon.txt"
THERAPIST_AUDIO = "audio_therapist.wav"
CLIENT_AUDIO = "audio_client.wav"
AUDIO_FILES = (THERAPIST_AUDIO, CLIENT_AUDIO)
_MARKER = ".recording.json"


def atomic_write_text(path: Path, text: str) -> None:
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


@contextmanager
def _locked(session_dir: Path) -> Iterator[None]:
    with open(session_dir / ".lock", "w") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        yield


def _infer_legacy(session_dir: Path) -> dict:
    """Describe a session folder written before session.json existed."""
    has_notes = (session_dir / NOTES_FILE).exists()
    return {
        "state": DONE if has_notes else UNPROCESSED,
        "couples": None,
        "legacy": True,
        "warnings": [],
        "attempts": 0,
    }


def load_session(session_dir: Path) -> dict:
    try:
        data = json.loads((session_dir / STATE_FILE).read_text(encoding="utf-8"))
    except FileNotFoundError:
        data = _infer_legacy(session_dir)
    except (ValueError, OSError):
        log.warning("Unreadable state file in session %s", session_dir.name)
        data = _infer_legacy(session_dir)
    data.setdefault("warnings", [])
    data.setdefault("attempts", 0)
    data["id"] = session_dir.name
    return data


def update_session(session_dir: Path, **fields) -> dict:
    with _locked(session_dir):
        data = load_session(session_dir)
        data.update(fields, updated_at=time.time())
        atomic_write_text(session_dir / STATE_FILE, json.dumps(data, indent=2))
    return data


def add_warning(session_dir: Path, message: str) -> None:
    with _locked(session_dir):
        data = load_session(session_dir)
        if message not in data["warnings"]:
            data["warnings"].append(message)
            atomic_write_text(session_dir / STATE_FILE, json.dumps(data, indent=2))


def create_session(root: Path, couples: bool) -> Path:
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    session_dir = root / stamp
    suffix = 1
    while True:
        try:
            session_dir.mkdir(mode=0o700)
            break
        except FileExistsError:
            suffix += 1
            session_dir = root / f"{stamp}_{suffix}"
    update_session(session_dir, state=RECORDING, couples=couples, created_at=time.time())
    return session_dir


def list_sessions(root: Path, limit: Optional[int] = None) -> list[tuple[Path, dict]]:
    """Sessions newest first."""
    try:
        dirs = [d for d in root.iterdir() if d.is_dir() and not d.name.startswith(".")]
    except OSError:
        return []
    dirs.sort(key=lambda d: d.name, reverse=True)
    if limit is not None:
        dirs = dirs[:limit]
    return [(d, load_session(d)) for d in dirs]


def audio_paths(session_dir: Path) -> list[Path]:
    return [session_dir / name for name in AUDIO_FILES if (session_dir / name).exists()]


def audio_bytes(session_dir: Path) -> int:
    return sum(path.stat().st_size for path in audio_paths(session_dir))


def delete_audio(session_dir: Path) -> int:
    """Delete a session's audio if its notes exist. Returns the bytes freed."""
    if not (session_dir / NOTES_FILE).exists():
        raise ValueError("Audio is only deleted after notes are saved")
    freed = 0
    for path in audio_paths(session_dir):
        freed += path.stat().st_size
        path.unlink(missing_ok=True)
    return freed


def mark_for_retry(session_dir: Path, couples: Optional[bool] = None) -> None:
    fields: dict = {"state": QUEUED, "error": None, "failed_stage": None, "attempts": 0}
    if couples is not None:
        fields["couples"] = couples
    update_session(session_dir, **fields)


# ── Active-recording marker ───────────────────────────────────────────────────
# The recorder holds this while it runs. The worker pauses while the marker's
# process is alive. A marker left by a dead process is ignored.

def write_marker(root: Path, session_id: str) -> None:
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    atomic_write_text(root / _MARKER, json.dumps({"session": session_id, "pid": os.getpid()}))


def clear_marker(root: Path) -> None:
    (root / _MARKER).unlink(missing_ok=True)


def active_recording(root: Path) -> Optional[str]:
    """The session id being recorded right now, or None."""
    try:
        marker = json.loads((root / _MARKER).read_text(encoding="utf-8"))
        os.kill(int(marker["pid"]), 0)
    except PermissionError:
        return str(marker["session"])
    except (OSError, ValueError, KeyError, TypeError):
        return None
    return str(marker["session"])
