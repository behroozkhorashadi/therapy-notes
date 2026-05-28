"""Tests for pure helper functions in setup_window.

These helpers are testable in isolation without a display or Qt event loop.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

from therapy_notes.ui.setup_window import _word_match, _do_update


# ── _word_match ───────────────────────────────────────────────────────────────

class TestWordMatch:
    def test_perfect_match(self):
        assert _word_match("hello world", "hello world") == 1.0

    def test_partial_match_half(self):
        ratio = _word_match("one two three four", "one two")
        assert ratio == pytest.approx(0.5)

    def test_no_match(self):
        assert _word_match("hello world", "foo bar") == 0.0

    def test_case_insensitive(self):
        assert _word_match("Hello World", "hello world") == 1.0

    def test_punctuation_stripped(self):
        assert _word_match("Hello, world.", "hello world") == 1.0

    def test_empty_expected_returns_zero(self):
        """Avoids division by zero; 0 expected words → 0.0."""
        assert _word_match("", "anything") == 0.0

    def test_extra_actual_words_do_not_lower_score(self):
        """All expected words present even if actual has extras."""
        ratio = _word_match("one two", "one two three four five")
        assert ratio == 1.0

    def test_order_does_not_matter(self):
        """Word match is set-based, not order-based."""
        assert _word_match("one two three", "three one two") == 1.0

    def test_duplicate_expected_words_counted_once(self):
        """Repeated words in expected collapse to a set."""
        ratio = _word_match("hello hello hello", "hello")
        assert ratio == 1.0


# ── _do_update repo_root ─────────────────────────────────────────────────────

class TestDoUpdateRepoRoot:
    """Verify that _do_update runs git in the actual project root."""

    def test_repo_root_has_git_dir(self):
        """parents[3] from setup_window.py resolves to the repo root."""
        from therapy_notes.ui import setup_window
        repo_root = Path(setup_window.__file__).parents[3]
        assert (repo_root / ".git").exists(), (
            f"Expected .git at {repo_root} — parents index may be wrong"
        )

    def test_git_pull_cwd_is_repo_root(self):
        """subprocess.run is called with cwd pointing inside a git repository."""
        from therapy_notes.ui.setup_window import _UpdateSignals

        captured: list[Path] = []

        def fake_run(cmd, **kwargs):
            captured.append(kwargs.get("cwd"))
            m = MagicMock()
            m.returncode = 0
            m.stdout = "Already up to date."
            m.stderr = ""
            return m

        sigs = _UpdateSignals()
        with patch("therapy_notes.ui.setup_window.subprocess.run", side_effect=fake_run):
            _do_update(sigs)

        assert captured, "subprocess.run was never called"
        cwd: Path = captured[0]
        assert (cwd / ".git").exists(), (
            f"git pull ran outside the repo — cwd was {cwd}"
        )

    def test_already_up_to_date_emits_ok(self):
        """'Already up to date.' stdout triggers the ok signal."""
        from therapy_notes.ui.setup_window import _UpdateSignals

        received: list[str] = []

        sigs = _UpdateSignals()
        sigs.ok.connect(received.append)

        def fake_run(cmd, **kwargs):
            m = MagicMock()
            m.returncode = 0
            m.stdout = "Already up to date."
            m.stderr = ""
            return m

        with patch("therapy_notes.ui.setup_window.subprocess.run", side_effect=fake_run):
            _do_update(sigs)

        assert received == ["Already up to date."]

    def test_git_pull_failure_emits_error(self):
        """A non-zero git exit code triggers the error signal."""
        from therapy_notes.ui.setup_window import _UpdateSignals

        errors: list[str] = []
        sigs = _UpdateSignals()
        sigs.error.connect(errors.append)

        def fake_run(cmd, **kwargs):
            m = MagicMock()
            m.returncode = 1
            m.stdout = ""
            m.stderr = "not a git repository"
            return m

        with patch("therapy_notes.ui.setup_window.subprocess.run", side_effect=fake_run):
            _do_update(sigs)

        assert errors and "not a git repository" in errors[0]
