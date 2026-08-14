from __future__ import annotations

import json
from pathlib import Path

import pytest

from opjax.pallas.laguna_depth_selection import select_depth


def _summary(path: Path, split: str = "calibration") -> None:
    payload = {
        "split": split,
        "sha256": "summary-sha",
        "cells": {
            "plain": {"requests": 18},
            "dflash-4": {
                "requests": 18,
                "wall_s": 10.0,
                "exact_plain_matches": 12,
                "result_sha256": "four",
            },
            "dflash-8": {
                "requests": 18,
                "wall_s": 9.0,
                "exact_plain_matches": 10,
                "result_sha256": "eight",
            },
            "dflash-12": {
                "requests": 18,
                "wall_s": 9.0,
                "exact_plain_matches": 11,
                "result_sha256": "twelve",
            },
        },
    }
    path.write_text(json.dumps(payload))


def test_selects_wall_then_matches_then_smaller_depth(tmp_path: Path) -> None:
    path = tmp_path / "summary.json"
    _summary(path)
    result = select_depth(path, "dflash")
    assert result["selected_depth"] == 12


def test_rejects_heldout_selection(tmp_path: Path) -> None:
    path = tmp_path / "summary.json"
    _summary(path, split="heldout")
    with pytest.raises(ValueError, match="REQUIRES_CALIBRATION"):
        select_depth(path, "dflash")
