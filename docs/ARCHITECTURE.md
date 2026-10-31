# Architecture

Three kinds of process share one folder on disk. A crash in any of them can't take
down the others or lose saved work.

| Process | Module | Job |
| --- | --- | --- |
| UI | `ui/app.py` | Shows state and sends commands. Imports no audio or AI libraries. |
| Recorder | `recorder_service.py`, `audio/recorder.py` | Writes audio to disk about once a second and reports problems. |
| Worker | `worker.py`, `pipeline.py` | Handles one session at a time at low priority, pauses during recording, exits when idle. |
| Health check | `health.py` | Short-lived self-check and diagnostic report. |

`session.py` is the source of truth. Each session folder holds `session.json`:

```
recording → queued → transcribing → anonymizing → generating → done
                         └───────────── any stage ─────────────→ failed (retry from the UI)
```

Each stage saves its result first, so a restart resumes where it stopped. Audio is
deleted only after notes are saved.

## Rules worth keeping

- Never log transcript, note or patient content.
- Only the worker writes processing states. The UI only changes finished or failed sessions.
- Anything that can hang (audio devices, models, network) stays out of the UI process.
- A cut-off, refused or empty note is an error, never a saved note.
- Risk fields are only completed from what was said. Silence is "Not assessed", never "Denied".

## Adding or updating models

- Local model picks by memory: `models.py`.
- A new speech engine: add a class to `transcription/engines.py` and register it in `create_engine`.
- Processing on another machine: implement the `LocalBackend` methods in `pipeline.py` as a remote backend.
