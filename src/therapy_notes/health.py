"""The app's self-check: python -m therapy_notes.health [--report]

Runs as its own short-lived process, so a hung audio driver can't freeze the UI and
the device list is always fresh. Prints JSON. Never reads session content.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import os
import platform
import re
import shutil
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

from . import session as store
from .config import config
from .logging_setup import log_path
from .models import detect_hardware, recommend_models

OK, WARN, URGENT = "ok", "warn", "urgent"
DISK_WARN_GB = 20
DISK_URGENT_GB = 2
GB = 1024**3


@dataclass
class Check:
    name: str
    level: str
    message: str
    fix: Optional[str] = None    # "cleanup", "setup", "sessions" or "report"


def check_disk(free_bytes: int) -> Check:
    free = free_bytes / GB
    if free < DISK_URGENT_GB:
        return Check("Disk space", URGENT, f"Only {free:.1f} GB free. The Mac may freeze and recordings can fail.", "cleanup")
    if free < DISK_WARN_GB:
        return Check("Disk space", WARN, f"{free:.0f} GB free. Under {DISK_WARN_GB} GB can slow the whole Mac.", "cleanup")
    return Check("Disk space", OK, f"{free:.0f} GB free.")


def check_stored_audio(sessions: list[tuple[Path, dict]]) -> Check:
    removable = sum(store.audio_bytes(d) for d, data in sessions if (d / store.NOTES_FILE).exists())
    total = sum(store.audio_bytes(d) for d, _ in sessions)
    if removable >= GB // 2:
        return Check("Stored audio", WARN, f"{removable / GB:.1f} GB of audio belongs to sessions that already have notes.", "cleanup")
    return Check("Stored audio", OK, f"{total / GB:.1f} GB of audio stored.")


def check_devices(cfg) -> Check:
    try:
        from .audio.devices import find_device
        import sounddevice as sd
        sd.query_devices()
    except Exception as exc:  # noqa: BLE001
        return Check("Audio devices", URGENT, f"Couldn't read audio devices: {exc}", "setup")
    if cfg.audio_primary_device and find_device(cfg.audio_primary_device, "input") is None:
        return Check("Audio devices", URGENT, f"Microphone '{cfg.audio_primary_device}' wasn't found.", "setup")
    if not cfg.audio_loopback_device:
        return Check("Audio devices", WARN, "Client audio isn't set up, so only the microphone will record.", "setup")
    if find_device(cfg.audio_loopback_device, "input") is None:
        return Check("Audio devices", WARN, f"'{cfg.audio_loopback_device}' wasn't found, so client audio won't record.", "setup")
    return Check("Audio devices", OK, "Microphone and client audio are available.")


def check_settings(cfg) -> Check:
    errors = cfg.validate()
    if errors:
        return Check("Settings", URGENT, errors[0], "setup")
    if cfg.ai_provider != "ollama" and importlib.util.find_spec("en_core_web_lg") is None:
        return Check("Settings", URGENT, "The anonymization model isn't installed.", "setup")
    return Check("Settings", OK, f"Using {cfg.ai_provider} for notes.")


def check_unfinished(sessions: list[tuple[Path, dict]], now: float) -> Check:
    failed = sum(1 for _, data in sessions if data.get("state") == store.FAILED)
    old_unprocessed = sum(1 for _, data in sessions if data.get("state") == store.UNPROCESSED)
    stuck = sum(
        1 for _, data in sessions
        if data.get("state") in store.IN_PROGRESS and now - data.get("updated_at", now) > 6 * 3600
    )
    if failed or stuck:
        return Check("Unfinished sessions", WARN, f"{failed + stuck} session(s) need a retry.", "sessions")
    if old_unprocessed:
        return Check("Unfinished sessions", WARN, f"{old_unprocessed} older session(s) have audio but no notes.", "sessions")
    return Check("Unfinished sessions", OK, "Nothing stuck.")


def check_crashes(now: float, reports_dir: Optional[Path] = None) -> Check:
    reports_dir = reports_dir or Path.home() / "Library" / "Logs" / "DiagnosticReports"
    try:
        recent = [p for p in reports_dir.glob("Python-*.ips") if now - p.stat().st_mtime < 7 * 86400]
    except OSError:
        recent = []
    if recent:
        return Check("Recent crashes", WARN, f"Python crashed {len(recent)} time(s) in the last week.", "report")
    return Check("Recent crashes", OK, "None in the last week.")


def run_checks(cfg=config) -> list[Check]:
    now = time.time()
    root = cfg.sessions_dir
    sessions = store.list_sessions(root)
    existing = root if root.exists() else Path.home()
    return [
        check_disk(shutil.disk_usage(existing).free),
        check_stored_audio(sessions),
        check_devices(cfg),
        check_settings(cfg),
        check_unfinished(sessions, now),
        check_crashes(now),
    ]


def _version(package: str) -> str:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


def _scrub(text: str) -> str:
    return re.sub(r"sk-[\w-]+", "[key]", text)


def build_report(cfg=config) -> str:
    """Text for troubleshooting from another Mac: no keys, names, transcripts or audio."""
    hw = detect_hardware()
    sessions = store.list_sessions(cfg.sessions_dir)
    states: dict[str, int] = {}
    for _, data in sessions:
        states[data.get("state", "?")] = states.get(data.get("state", "?"), 0) + 1
    lines = [
        "Therapy Notes diagnostic report",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M')}",
        f"macOS {platform.mac_ver()[0]}, {platform.machine()}, {hw.ram_gb} GB RAM, Python {platform.python_version()}",
        f"Recommended models: {recommend_models(hw).summary}",
        "",
        "Settings:",
        f"  AI provider: {cfg.ai_provider}",
        f"  API key set: anthropic={bool(cfg.anthropic_api_key)}, openai={bool(cfg.openai_api_key)}",
        f"  Models: anthropic={cfg.ai_model_anthropic}, openai={cfg.ai_model_openai}, ollama={cfg.ollama_model}",
        f"  Speech: engine={cfg.asr_engine}, whisper={cfg.whisper_model}",
        f"  Delete audio after notes: {cfg.delete_audio_after_notes}",
        f"  Client audio configured: {bool(cfg.audio_loopback_device)}",
        "",
        "Packages: " + ", ".join(
            f"{p} {_version(p)}" for p in ("parakeet-mlx", "faster-whisper", "sounddevice", "PyQt6", "anthropic", "openai")
        ),
        "",
        f"Sessions: {len(sessions)} {states}",
        f"Stored audio: {sum(store.audio_bytes(d) for d, _ in sessions) / GB:.1f} GB",
        "",
        "Checks:",
    ]
    lines += [f"  [{c.level}] {c.name}: {c.message}" for c in run_checks(cfg)]
    lines += ["", "Failed sessions:"]
    failed = [(data["id"], data.get("failed_stage"), data.get("error")) for _, data in sessions if data.get("state") == store.FAILED]
    lines += [f"  {i} at {stage}: {error}" for i, stage, error in failed] or ["  none"]
    lines += ["", "Recent log:"]
    try:
        lines += log_path().read_text(encoding="utf-8", errors="replace").splitlines()[-60:]
    except OSError:
        lines.append("  (no log yet)")
    return _scrub("\n".join(lines))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", action="store_true")
    args = parser.parse_args()
    if args.report:
        print(build_report())
    else:
        print(json.dumps([asdict(check) for check in run_checks()]))
    sys.stdout.flush()
    os._exit(0)     # don't wait on PortAudio shutdown


if __name__ == "__main__":
    raise SystemExit(main())
