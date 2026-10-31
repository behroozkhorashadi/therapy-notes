"""Session store: atomic state, legacy folders, and the audio-deletion guard."""
from __future__ import annotations

import os
import subprocess
import sys

import pytest

from therapy_notes import session as store


def test_create_and_update(tmp_path):
    d = store.create_session(tmp_path, couples=True)
    assert store.load_session(d)["state"] == store.RECORDING
    store.update_session(d, state=store.QUEUED)
    data = store.load_session(d)
    assert data["state"] == store.QUEUED and data["couples"] is True and data["id"] == d.name


def test_same_second_sessions_get_separate_folders(tmp_path):
    assert store.create_session(tmp_path, False) != store.create_session(tmp_path, False)


def test_legacy_folder_with_notes_is_done(tmp_path):
    d = tmp_path / "2026-05-01_10-00-00"
    d.mkdir()
    (d / store.NOTES_FILE).write_text("n")
    assert store.load_session(d)["state"] == store.DONE


def test_legacy_audio_without_notes_is_not_auto_queued(tmp_path):
    d = tmp_path / "2026-05-01_10-00-00"
    d.mkdir()
    (d / store.THERAPIST_AUDIO).write_bytes(b"x")
    state = store.load_session(d)["state"]
    assert state == store.UNPROCESSED and state not in store.PENDING


def test_corrupt_state_file_does_not_raise(tmp_path):
    d = tmp_path / "s"
    d.mkdir()
    (d / store.STATE_FILE).write_text("{not json")
    assert store.load_session(d)["state"] == store.UNPROCESSED


def test_list_newest_first_and_skips_hidden(tmp_path):
    for name in ("2026-01-01_00-00-00", "2026-02-01_00-00-00", ".hidden"):
        (tmp_path / name).mkdir()
    assert [d.name for d, _ in store.list_sessions(tmp_path)] == ["2026-02-01_00-00-00", "2026-01-01_00-00-00"]


def test_delete_audio_refuses_without_notes(tmp_path):
    (tmp_path / store.THERAPIST_AUDIO).write_bytes(b"x")
    with pytest.raises(ValueError):
        store.delete_audio(tmp_path)
    assert (tmp_path / store.THERAPIST_AUDIO).exists()


def test_delete_audio_with_notes(tmp_path):
    (tmp_path / store.NOTES_FILE).write_text("n")
    (tmp_path / store.THERAPIST_AUDIO).write_bytes(b"abc")
    (tmp_path / store.CLIENT_AUDIO).write_bytes(b"de")
    assert store.delete_audio(tmp_path) == 5
    assert store.audio_paths(tmp_path) == []


def test_atomic_write_leaves_old_file_on_failure(tmp_path, monkeypatch):
    target = tmp_path / "f.txt"
    target.write_text("old")
    monkeypatch.setattr(os, "fsync", lambda fd: (_ for _ in ()).throw(OSError(28, "No space left")))
    with pytest.raises(OSError):
        store.atomic_write_text(target, "new")
    assert target.read_text() == "old"
    assert [p.name for p in tmp_path.iterdir()] == ["f.txt"]


def test_marker_live_and_stale(tmp_path):
    assert store.active_recording(tmp_path) is None
    store.write_marker(tmp_path, "abc")
    assert store.active_recording(tmp_path) == "abc"
    child = subprocess.Popen([sys.executable, "-c", "pass"])
    child.wait()
    (tmp_path / store._MARKER).write_text(f'{{"session": "abc", "pid": {child.pid}}}')
    assert store.active_recording(tmp_path) is None
    store.clear_marker(tmp_path)
    assert store.active_recording(tmp_path) is None


def test_retry_clears_failure(tmp_path):
    store.update_session(tmp_path, state=store.FAILED, error="boom", attempts=3)
    store.mark_for_retry(tmp_path, couples=True)
    data = store.load_session(tmp_path)
    assert data["state"] == store.QUEUED and data["error"] is None and data["attempts"] == 0 and data["couples"] is True
