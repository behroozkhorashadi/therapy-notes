"""Tests for the audio device finder."""
from __future__ import annotations

from unittest.mock import patch

import pytest

from therapy_notes.audio.devices import find_device


def _dev(name: str, inputs: int = 0, outputs: int = 0) -> dict:
    """Build a minimal device dict matching sounddevice's query_devices format."""
    return {"name": name, "max_input_channels": inputs, "max_output_channels": outputs}


def _patch(devices):
    return patch("therapy_notes.audio.devices.sd.query_devices", return_value=devices)


class TestFindDevice:
    def test_finds_input_device(self):
        devs = [_dev("BlackHole 2ch", inputs=2), _dev("MacBook Speakers", outputs=2)]
        with _patch(devs):
            assert find_device("BlackHole 2ch", kind="input") == 0

    def test_finds_output_device(self):
        devs = [_dev("BlackHole 2ch", inputs=2), _dev("Therapy Output", outputs=2)]
        with _patch(devs):
            assert find_device("Therapy Output", kind="output") == 1

    def test_returns_correct_index(self):
        devs = [
            _dev("Device A", inputs=1),
            _dev("Device B", inputs=1),
            _dev("BlackHole 2ch", inputs=2),
        ]
        with _patch(devs):
            assert find_device("BlackHole 2ch", kind="input") == 2

    def test_case_insensitive(self):
        devs = [_dev("BlackHole 2ch", inputs=2)]
        with _patch(devs):
            assert find_device("blackhole 2ch", kind="input") == 0
            assert find_device("BLACKHOLE 2CH", kind="input") == 0

    def test_partial_name_match(self):
        devs = [_dev("BlackHole 2ch (Virtual)", inputs=2)]
        with _patch(devs):
            assert find_device("BlackHole", kind="input") == 0

    def test_returns_none_when_no_match(self):
        devs = [_dev("Built-in Microphone", inputs=1)]
        with _patch(devs):
            assert find_device("BlackHole 2ch", kind="input") is None

    def test_input_only_device_not_returned_for_output_search(self):
        devs = [_dev("BlackHole 2ch", inputs=2, outputs=0)]
        with _patch(devs):
            assert find_device("BlackHole 2ch", kind="output") is None

    def test_output_only_device_not_returned_for_input_search(self):
        devs = [_dev("MacBook Speakers", inputs=0, outputs=2)]
        with _patch(devs):
            assert find_device("MacBook Speakers", kind="input") is None

    def test_empty_device_list(self):
        with _patch([]):
            assert find_device("anything", kind="input") is None

    def test_device_with_both_channels_satisfies_either_kind(self):
        """A device with input and output channels matches both searches."""
        devs = [_dev("Aggregate Device", inputs=2, outputs=2)]
        with _patch(devs):
            assert find_device("Aggregate", kind="input") == 0
            assert find_device("Aggregate", kind="output") == 0
