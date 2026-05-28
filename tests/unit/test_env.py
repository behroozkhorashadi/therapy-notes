"""Tests for the save_env_value utility."""
from __future__ import annotations

from pathlib import Path

import pytest

from therapy_notes.utils.env import save_env_value


class TestSaveEnvValue:
    def test_creates_file_when_missing(self, tmp_path: Path):
        env = tmp_path / ".env"
        save_env_value("FOO", "bar", env_path=env)
        assert env.exists()
        assert "FOO=bar" in env.read_text()

    def test_replaces_existing_key(self, tmp_path: Path):
        env = tmp_path / ".env"
        env.write_text("FOO=old\nBAR=keep\n")
        save_env_value("FOO", "new", env_path=env)
        content = env.read_text()
        assert "FOO=new" in content
        assert "FOO=old" not in content

    def test_does_not_clobber_other_keys(self, tmp_path: Path):
        env = tmp_path / ".env"
        env.write_text("KEEP=this\nFOO=old\nALSO=keep\n")
        save_env_value("FOO", "new", env_path=env)
        content = env.read_text()
        assert "KEEP=this" in content
        assert "ALSO=keep" in content

    def test_appends_new_key(self, tmp_path: Path):
        env = tmp_path / ".env"
        env.write_text("EXISTING=yes\n")
        save_env_value("NEW_KEY", "value", env_path=env)
        content = env.read_text()
        assert "EXISTING=yes" in content
        assert "NEW_KEY=value" in content

    def test_no_duplicate_key_on_double_write(self, tmp_path: Path):
        """Calling save_env_value twice replaces, not appends."""
        import re
        env = tmp_path / ".env"
        save_env_value("MYKEY", "first", env_path=env)
        save_env_value("MYKEY", "second", env_path=env)
        content = env.read_text()
        # Use anchored regex so we don't match substring occurrences like OTHER_MYKEY=
        matches = re.findall(r"^MYKEY=", content, re.MULTILINE)
        assert len(matches) == 1, f"Expected exactly one MYKEY= line, found: {matches}"
        assert "MYKEY=second" in content

    def test_creates_from_example_when_env_missing(self, tmp_path: Path, monkeypatch):
        """If .env is absent but .env.example exists in cwd, it seeds from example."""
        # save_env_value resolves .env.example relative to cwd, so we chdir first.
        monkeypatch.chdir(tmp_path)
        example = tmp_path / ".env.example"
        example.write_text("# Example config\nSAMPLE=default\n")
        env = tmp_path / ".env"
        save_env_value("NEW", "val", env_path=env)
        content = env.read_text()
        assert "# Example config" in content
        assert "SAMPLE=default" in content
        assert "NEW=val" in content

    def test_creates_empty_when_no_example(self, tmp_path: Path):
        """If neither .env nor .env.example exist, creates a fresh .env."""
        env = tmp_path / ".env"
        save_env_value("KEY", "val", env_path=env)
        assert env.exists()
        assert "KEY=val" in env.read_text()

    def test_file_without_trailing_newline(self, tmp_path: Path):
        """A file with no trailing newline still gets new keys on their own line."""
        env = tmp_path / ".env"
        env.write_text("EXISTING=yes")   # no trailing newline
        save_env_value("NEW", "val", env_path=env)
        lines = env.read_text().splitlines()
        assert any(line == "EXISTING=yes" for line in lines)
        assert any(line == "NEW=val" for line in lines)

    def test_partial_key_name_not_replaced(self, tmp_path: Path):
        """FOO= must not be matched when the file contains FOOBAR=."""
        env = tmp_path / ".env"
        env.write_text("FOOBAR=keep\n")
        save_env_value("FOO", "new", env_path=env)
        content = env.read_text()
        assert "FOOBAR=keep" in content
        assert "FOO=new" in content
