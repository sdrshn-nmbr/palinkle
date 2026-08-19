from __future__ import annotations

import json
from pathlib import Path

import pytest

from opjax.pallas.laguna_depth_selection import select_depth
from opjax.pallas.laguna_dspark_conformance import canonical_sha256


def _summary(path: Path, split: str = "calibration") -> None:
    cells = {
        "plain": {"requests": 18},
        "dspark-adaptive": {
            "requests": 18,
            "wall_s": 11.0,
            "exact_plain_matches": 12,
            "result_sha256": "dspark-adaptive",
        },
    }
    common_cells = {}
    ratios = {4: 1.1, 8: 1.2, 12: 1.3, 15: 1.25}
    for arm in ("dflash", "dspark"):
        for depth in (4, 8, 12, 15):
            cell = f"{arm}-{depth}"
            result_sha = f"{arm}-{depth}-result"
            cells[cell] = {
                "requests": 18,
                "wall_s": 12.0 - depth / 10,
                "exact_plain_matches": 12,
                "result_sha256": result_sha,
            }
            common_cells[cell] = {
                "plain_over_cell_latency": {
                    "ratio": ratios[depth],
                    "ci95_low": ratios[depth] - 0.05,
                    "ci95_high": ratios[depth] + 0.05,
                    "prompts": 10,
                },
                "result_sha256": result_sha,
            }
    payload = {
        "split": split,
        "cells": cells,
        "depth_common_exact": {
            arm: {
                "prompts": 10,
                "prompt_ids_sha256": "prompt-ids",
                "cells": {
                    cell: value
                    for cell, value in common_cells.items()
                    if cell.startswith(f"{arm}-")
                },
            }
            for arm in ("dflash", "dspark")
        },
    }
    payload["sha256"] = canonical_sha256(payload)
    path.write_text(json.dumps(payload))


def test_selects_common_exact_paired_latency(tmp_path: Path) -> None:
    path = tmp_path / "summary.json"
    _summary(path)
    result = select_depth(path, "dflash")
    assert result["selected_depth"] == 12


def test_rejects_heldout_selection(tmp_path: Path) -> None:
    path = tmp_path / "summary.json"
    _summary(path, split="heldout")
    with pytest.raises(ValueError, match="REQUIRES_CALIBRATION"):
        select_depth(path, "dflash")


def test_rejects_slower_adaptive_schedule(tmp_path: Path) -> None:
    path = tmp_path / "summary.json"
    _summary(path)
    result = select_depth(path, "dspark")
    assert result["selected_depth"] == 12
    assert result["adaptive"] is None


def test_rejects_mutated_summary(tmp_path: Path) -> None:
    path = tmp_path / "summary.json"
    _summary(path)
    payload = json.loads(path.read_text())
    payload["cells"]["dflash-12"]["wall_s"] = 0.1
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="SUMMARY_HASH_MISMATCH"):
        select_depth(path, "dflash")


def test_rejects_nonfinite_ratio(tmp_path: Path) -> None:
    path = tmp_path / "summary.json"
    _summary(path)
    payload = json.loads(path.read_text())
    payload["depth_common_exact"]["dflash"]["cells"]["dflash-12"][
        "plain_over_cell_latency"
    ]["ratio"] = float("nan")
    payload["sha256"] = canonical_sha256(
        {key: value for key, value in payload.items() if key != "sha256"}
    )
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="COMMON_EXACT_INVALID"):
        select_depth(path, "dflash")
