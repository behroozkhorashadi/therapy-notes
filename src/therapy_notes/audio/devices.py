from __future__ import annotations

from typing import Optional

import sounddevice as sd


def find_device(name: str, kind: str = "input") -> Optional[int]:
    """Return the device index whose name contains *name* (case-insensitive).

    kind: "input" or "output" — filters by channel availability.
    Returns None if no match is found.
    """
    for i, dev in enumerate(sd.query_devices()):
        if name.lower() in dev["name"].lower():
            if kind == "input" and dev["max_input_channels"] > 0:
                return i
            if kind == "output" and dev["max_output_channels"] > 0:
                return i
    return None
