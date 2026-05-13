from __future__ import annotations

import re
from pathlib import Path

from dotenv import load_dotenv


def save_env_value(key: str, value: str, env_path: Path | None = None) -> None:
    """Write key=value to the .env file (creates or replaces) and reload dotenv."""
    if env_path is None:
        env_path = Path(".env")
    if not env_path.exists():
        example = Path(".env.example")
        env_path.write_text(example.read_text() if example.exists() else "")
    content = env_path.read_text()
    replacement = f"{key}={value}"
    if re.search(rf"^{key}=", content, re.MULTILINE):
        content = re.sub(rf"^{key}=.*$", replacement, content, flags=re.MULTILINE)
    else:
        content = content.rstrip("\n") + f"\n{replacement}\n"
    env_path.write_text(content)
    load_dotenv(override=True)
