from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from opjax.pallas.laguna_dspark_conformance import canonical_sha256


def select_depth(summary_path: Path, arm: str) -> dict[str, Any]:
    if arm not in {"dflash", "dspark"}:
        raise ValueError(f"LAGUNA_DEPTH_ARM_INVALID:{arm}")
    summary = json.loads(summary_path.read_text())
    expected_summary_sha256 = canonical_sha256(
        {key: value for key, value in summary.items() if key != "sha256"}
    )
    if summary.get("sha256") != expected_summary_sha256:
        raise ValueError("LAGUNA_DEPTH_SUMMARY_HASH_MISMATCH")
    if summary.get("split") != "calibration":
        raise ValueError("LAGUNA_DEPTH_REQUIRES_CALIBRATION")
    prefix = f"{arm}-"
    common = summary.get("depth_common_exact", {}).get(arm)
    if not isinstance(common, dict) or int(common.get("prompts", 0)) <= 0:
        raise ValueError(f"LAGUNA_DEPTH_COMMON_EXACT_MISSING:{arm}")
    common_cells = common.get("cells")
    if not isinstance(common_cells, dict):
        raise ValueError(f"LAGUNA_DEPTH_COMMON_EXACT_INVALID:{arm}")
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
                "common_exact_latency": common_cells[cell][
                    "plain_over_cell_latency"
                ],
            }
        )
    if not rows:
        raise ValueError(f"LAGUNA_DEPTH_RESULTS_MISSING:{arm}")
    request_counts = {row["requests"] for row in rows}
    if len(request_counts) != 1:
        raise ValueError(f"LAGUNA_DEPTH_REQUEST_COUNT_MISMATCH:{arm}")
    expected_cells = {f"{arm}-{depth}" for depth in (4, 8, 12, 15)}
    if {row["cell"] for row in rows} != expected_cells or set(common_cells) != expected_cells:
        raise ValueError(f"LAGUNA_DEPTH_CELL_SET_INVALID:{arm}")
    for row in rows:
        latency = row["common_exact_latency"]
        ratio = latency.get("ratio")
        interval = (latency.get("ci95_low"), latency.get("ci95_high"))
        if (
            latency.get("prompts") != common["prompts"]
            or not isinstance(ratio, (int, float))
            or not math.isfinite(float(ratio))
            or float(ratio) <= 0
            or any(
                not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) <= 0
                for value in interval
            )
            or float(interval[0]) > float(interval[1])
            or row["result_sha256"] != common_cells[row["cell"]]["result_sha256"]
        ):
            raise ValueError(f"LAGUNA_DEPTH_COMMON_EXACT_INVALID:{row['cell']}")
    selected = max(
        rows,
        key=lambda row: (
            float(row["common_exact_latency"]["ratio"]),
            -row["wall_s"],
            -row["depth"],
        ),
    )
    result = {
        "schema_version": 1,
        "arm": arm,
        "policy": (
            "maximize paired plain-over-cell latency on prompts whose token IDs "
            "match plain at every fixed depth; tie break lower fixed-request wall "
            "time, then smaller proposal depth"
        ),
        "common_exact_prompts": common["prompts"],
        "common_exact_prompt_ids_sha256": common["prompt_ids_sha256"],
        "summary_sha256": summary["sha256"],
        "selected_depth": selected["depth"],
        "selected": selected,
        "candidates": sorted(rows, key=lambda row: row["depth"]),
        "adaptive": None,
    }
    result["sha256"] = canonical_sha256(result)
    return result
