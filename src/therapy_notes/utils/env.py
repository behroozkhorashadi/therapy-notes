from __future__ import annotations

import os
import re
from pathlib import Path

from dotenv import load_dotenv


def save_env_value(key: str, value: str, env_path: Path | None = None) -> None:
    """Write key=value to the .env file (creates or replaces) and reload it."""
    if env_path is None:
        from ..config import ENV_PATH
        env_path = ENV_PATH
    if not env_path.exists():
        example = env_path.parent / ".env.example"
        env_path.write_text(example.read_text() if example.exists() else "")
    content = env_path.read_text()
    replacement = f"{key}={value}"
    if re.search(rf"^{key}=", content, re.MULTILINE):
        content = re.sub(rf"^{key}=.*$", lambda _: replacement, content, flags=re.MULTILINE)
    else:
        content = content.rstrip("\n") + f"\n{replacement}\n"
    env_path.write_text(content)
    os.chmod(env_path, 0o600)     # holds API keys
    load_dotenv(env_path, override=True)
