"""Health checks and the diagnostic report."""
from __future__ import annotations

import os
import time

from therapy_notes import health, session as store

GB = 1024**3


def test_disk_levels():
    assert health.check_disk(50 * GB).level == health.OK
    assert health.check_disk(10 * GB).level == health.WARN
    urgent = health.check_disk(1 * GB)
    assert urgent.level == health.URGENT and urgent.fix == "cleanup"


def _session(root, name, audio=0, notes=False):
    d = root / name
    d.mkdir()
    if audio:
        with open(d / store.THERAPIST_AUDIO, "wb") as handle:
            handle.truncate(audio)
    if notes:
        (d / store.NOTES_FILE).write_text("SECRET NOTE TEXT")
    return d


def test_stored_audio_counts_only_finished_sessions_as_removable(tmp_path):
    _session(tmp_path, "a", audio=GB, notes=True)
    assert health.check_stored_audio(store.list_sessions(tmp_path)).level == health.WARN


def test_audio_without_notes_is_not_called_removable(tmp_path):
    _session(tmp_path, "a", audio=GB)
    assert health.check_stored_audio(store.list_sessions(tmp_path)).level == health.OK


def test_unfinished(tmp_path):
    now = time.time()
    assert health.check_unfinished([], now).level == health.OK
    d = _session(tmp_path, "a")
    store.update_session(d, state=store.FAILED)
    check = health.check_unfinished(store.list_sessions(tmp_path), now)
    assert check.level == health.WARN and check.fix == "sessions"


def test_stuck_job_is_flagged_but_active_job_is_not(tmp_path):
    d = _session(tmp_path, "a")
    store.update_session(d, state=store.TRANSCRIBING)
    now = time.time()
    assert health.check_unfinished(store.list_sessions(tmp_path), now).level == health.OK
    assert health.check_unfinished(store.list_sessions(tmp_path), now + 7 * 3600).level == health.WARN


def test_settings_flags_placeholder_key(mock_config):
    check = health.check_settings(mock_config(ANTHROPIC_API_KEY="your_anthropic_key_here"))
    assert check.level == health.URGENT


def test_crash_reports(tmp_path):
    now = time.time()
    assert health.check_crashes(now, tmp_path).level == health.OK
    (tmp_path / "Python-2026.ips").write_text("x")
    assert health.check_crashes(now, tmp_path).level == health.WARN
    os.utime(tmp_path / "Python-2026.ips", (now - 30 * 86400, now - 30 * 86400))
    assert health.check_crashes(now, tmp_path).level == health.OK


def test_report_has_no_keys_or_session_content(tmp_path, monkeypatch, mock_config):
    monkeypatch.setenv("THERAPY_NOTES_DATA_DIR", str(tmp_path / "data"))
    root = tmp_path / "sessions"
    root.mkdir()
    cfg = mock_config(ANTHROPIC_API_KEY="sk-ant-supersecret", SESSIONS_DIR=str(root))
    d = _session(root, "2026-01-01_00-00-00", notes=True)
    (d / store.RAW_TRANSCRIPT).write_text("Client: my name is SECRET NAME")
    failed = _session(root, "2026-01-02_00-00-00")
    store.update_session(failed, state=store.FAILED, failed_stage="generating", error="bad key sk-ant-leaky")
    log = tmp_path / "data" / "logs"
    log.mkdir(parents=True)
    (log / "therapy-notes.log").write_text("error using sk-ant-inlog\n")
    monkeypatch.setattr(health, "check_devices", lambda c: health.Check("Audio devices", health.OK, "fine"))
    report = health.build_report(cfg)
    for secret in ("supersecret", "SECRET NAME", "SECRET NOTE", "sk-ant-leaky", "sk-ant-inlog"):
        assert secret not in report
    assert "2026-01-02_00-00-00" in report and "API key set: anthropic=True" in report
