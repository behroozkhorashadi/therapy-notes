"""Pipeline stages: ordering, resume, and when data may be deleted."""
from __future__ import annotations

import pytest

from therapy_notes import pipeline, session as store
from therapy_notes.config import Config


class FakeBackend:
    def __init__(self, transcript="Therapist: Hi\n\nClient: Hello", client_speech=True, fail=None):
        self.transcript, self.client_speech, self.fail = transcript, client_speech, fail
        self.calls = []
        self.notes_input = None
        self.audio_at_notes = None

    def _maybe_fail(self, stage):
        self.calls.append(stage)
        if self.fail == stage:
            raise RuntimeError(f"{stage} failed")

    def transcribe(self, therapist, client):
        self._maybe_fail("transcribe")
        return self.transcript, self.client_speech

    def anonymize(self, text):
        self._maybe_fail("anonymize")
        return "ANON " + text

    def generate_notes(self, text, couples, anonymized):
        self._maybe_fail("notes")
        self.notes_input = (text, couples, anonymized)
        return "THE NOTES"


@pytest.fixture
def session(tmp_path, monkeypatch, mock_config):
    cfg = mock_config(DELETE_AUDIO_AFTER_NOTES="true")
    monkeypatch.setattr(pipeline, "config", cfg)
    d = store.create_session(tmp_path / "root", couples=True)
    (d / store.THERAPIST_AUDIO).write_bytes(b"a")
    (d / store.CLIENT_AUDIO).write_bytes(b"b")
    store.update_session(d, state=store.QUEUED)
    return d


def test_full_run_saves_notes_then_deletes_audio(session):
    backend = FakeBackend()
    pipeline.process_session(session, backend)
    assert backend.calls == ["transcribe", "anonymize", "notes"]
    assert (session / store.NOTES_FILE).read_text() == "THE NOTES"
    assert store.audio_paths(session) == []
    assert store.load_session(session)["state"] == store.DONE


def test_cloud_provider_receives_scrubbed_text_only(session):
    backend = FakeBackend()
    pipeline.process_session(session, backend)
    text, couples, anonymized = backend.notes_input
    assert text.startswith("ANON ") and couples is True and anonymized is True


def test_anonymizer_failure_never_sends_raw_text(session):
    backend = FakeBackend(fail="anonymize")
    with pytest.raises(RuntimeError):
        pipeline.process_session(session, backend)
    assert backend.notes_input is None
    assert "notes" not in backend.calls


def test_local_provider_skips_scrubbing(session, monkeypatch, mock_config):
    monkeypatch.setattr(pipeline, "config", mock_config(AI_PROVIDER="ollama"))
    backend = FakeBackend()
    pipeline.process_session(session, backend)
    assert "anonymize" not in backend.calls
    assert backend.notes_input[2] is False


@pytest.mark.parametrize("stage", ["transcribe", "anonymize", "notes"])
def test_failure_keeps_audio(session, stage):
    with pytest.raises(RuntimeError):
        pipeline.process_session(session, FakeBackend(fail=stage))
    assert len(store.audio_paths(session)) == 2
    assert not (session / store.NOTES_FILE).exists()


def test_resume_skips_finished_stages(session):
    with pytest.raises(RuntimeError):
        pipeline.process_session(session, FakeBackend(fail="notes"))
    second = FakeBackend()
    pipeline.process_session(session, second)
    assert second.calls == ["notes"]
    assert store.load_session(session)["state"] == store.DONE


def test_empty_transcript_fails_instead_of_inventing_a_note(session):
    backend = FakeBackend(transcript="   ")
    with pytest.raises(pipeline.PipelineError, match="No speech"):
        pipeline.process_session(session, backend)
    assert "notes" not in backend.calls
    assert len(store.audio_paths(session)) == 2


def test_silent_client_warns(session):
    pipeline.process_session(session, FakeBackend(transcript="Therapist: Hi", client_speech=False))
    assert any("No client speech" in w for w in store.load_session(session)["warnings"])


def test_mic_only_warns(session):
    (session / store.CLIENT_AUDIO).unlink()
    pipeline.process_session(session, FakeBackend(transcript="Hi", client_speech=False))
    assert any("Only the microphone" in w for w in store.load_session(session)["warnings"])


def test_missing_audio_fails(session):
    (session / store.THERAPIST_AUDIO).unlink()
    with pytest.raises(pipeline.PipelineError, match="missing"):
        pipeline.process_session(session, FakeBackend())


def test_keeps_audio_when_setting_is_off(session, monkeypatch, mock_config):
    monkeypatch.setattr(pipeline, "config", mock_config(DELETE_AUDIO_AFTER_NOTES="false"))
    pipeline.process_session(session, FakeBackend())
    assert len(store.audio_paths(session)) == 2
