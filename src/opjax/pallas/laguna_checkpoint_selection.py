from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from opjax.pallas.laguna_dspark_conformance import canonical_sha256


def select_checkpoint(root: Path, arm: str) -> dict[str, Any]:
    rows = []
    arm_root = root / arm / "raw"
    for path in sorted(arm_root.glob("step_*/result.json")):
        payload = json.loads(path.read_text())
        if payload.get("arm") != arm or payload.get("split") != "calibration":
            raise ValueError(f"LAGUNA_CHECKPOINT_RESULT_INVALID:{path}")
        rows.append(
            {
                "step": int(payload["step"]),
                "probabilistic_tau": float(payload["probabilistic_tau"]),
                "greedy_tau": float(payload["greedy_tau"]),
                "cross_entropy": float(payload["loss"]["cross_entropy"]),
                "result_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    if not rows:
        raise ValueError(f"LAGUNA_CHECKPOINT_RESULTS_MISSING:{arm}")
    selected = max(
        rows,
        key=lambda row: (
            row["probabilistic_tau"],
            -row["cross_entropy"],
            -row["step"],
        ),
    )
    result = {
        "schema_version": 1,
        "arm": arm,
        "policy": (
            "maximize calibration probabilistic_tau; tie break lower cross_entropy; "
            "then earlier step"
        ),
        "selected_step": selected["step"],
        "selected": selected,
        "candidates": rows,
    }
    result["sha256"] = canonical_sha256(result)
    return result
