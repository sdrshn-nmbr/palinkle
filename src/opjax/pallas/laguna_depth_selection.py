from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from opjax.pallas.laguna_dspark_conformance import canonical_sha256


def select_depth(summary_path: Path, arm: str) -> dict[str, Any]:
    if arm not in {"dflash", "dspark"}:
        raise ValueError(f"LAGUNA_DEPTH_ARM_INVALID:{arm}")
    summary = json.loads(summary_path.read_text())
    if summary.get("split") != "calibration":
        raise ValueError("LAGUNA_DEPTH_REQUIRES_CALIBRATION")
    prefix = f"{arm}-"
    rows = []
    for cell, payload in summary["cells"].items():
        suffix = cell.removeprefix(prefix)
        if not cell.startswith(prefix) or not suffix.isdigit():
            continue
        if int(payload["requests"]) <= 0:
            raise ValueError(f"LAGUNA_DEPTH_CELL_EMPTY:{cell}")
        rows.append(
            {
                "cell": cell,
                "depth": int(suffix),
                "requests": int(payload["requests"]),
                "wall_s": float(payload["wall_s"]),
                "exact_plain_matches": int(payload["exact_plain_matches"]),
                "result_sha256": payload["result_sha256"],
            }
        )
    if not rows:
        raise ValueError(f"LAGUNA_DEPTH_RESULTS_MISSING:{arm}")
    request_counts = {row["requests"] for row in rows}
    if len(request_counts) != 1:
        raise ValueError(f"LAGUNA_DEPTH_REQUEST_COUNT_MISMATCH:{arm}")
    selected = min(
        rows,
        key=lambda row: (
            row["wall_s"],
            -row["exact_plain_matches"],
            row["depth"],
        ),
    )
    adaptive = summary["cells"].get("dspark-adaptive") if arm == "dspark" else None
    adaptive_decision = None
    if adaptive is not None:
        adaptive_decision = {
            "cell": "dspark-adaptive",
            "requests": int(adaptive["requests"]),
            "wall_s": float(adaptive["wall_s"]),
            "exact_plain_matches": int(adaptive["exact_plain_matches"]),
            "result_sha256": adaptive["result_sha256"],
            "admitted": (
                int(adaptive["requests"]) == selected["requests"]
                and float(adaptive["wall_s"]) < selected["wall_s"]
                and int(adaptive["exact_plain_matches"])
                >= selected["exact_plain_matches"]
            ),
            "policy": (
                "admit only when fixed-request wall time improves without fewer "
                "exact plain matches"
            ),
        }
    result = {
        "schema_version": 1,
        "arm": arm,
        "policy": (
            "minimize calibration fixed-request wall time; tie break more exact "
            "plain matches; then smaller proposal depth"
        ),
        "summary_sha256": summary["sha256"],
        "selected_depth": selected["depth"],
        "selected": selected,
        "candidates": sorted(rows, key=lambda row: row["depth"]),
        "adaptive": adaptive_decision,
    }
    result["sha256"] = canonical_sha256(result)
    return result
