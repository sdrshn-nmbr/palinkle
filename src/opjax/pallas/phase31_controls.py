"""Positive Pallas controls for representative Phase 3.1 operation families."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from opjax.pallas.g42_harness import G42HarnessError, canonical_sha256, file_sha256


CONTROL_FAMILIES = {
    "1p_Flash_Attention": "attention",
    "8p_GEMM": "matmul",
    "12p_RMSNorm": "normalization",
    "36k_Matmul_Sigmoid_Sum": "compound_matmul_reduction",
}


def _git_environment() -> dict[str, str]:
    return {
        **os.environ,
        "GIT_AUTHOR_NAME": "opjax-harness",
        "GIT_AUTHOR_EMAIL": "harness@opjax.invalid",
        "GIT_COMMITTER_NAME": "opjax-harness",
        "GIT_COMMITTER_EMAIL": "harness@opjax.invalid",
    }


def build_patches(
    *, release_root: Path, control_root: Path, out_dir: Path
) -> dict[str, Any]:
    if out_dir.exists():
        raise G42HarnessError(f"PHASE31_CONTROL_OUTPUT_EXISTS:{out_dir}")
    release = json.loads((release_root / "manifest.json").read_text())
    tasks = {task["task_id"]: task for task in release["tasks"]}
    records = []
    out_dir.mkdir(parents=True)
    for task_id, family in CONTROL_FAMILIES.items():
        source = control_root / f"{task_id}.py"
        task_root = release_root / tasks[task_id]["path"]
        with tempfile.TemporaryDirectory(prefix="opjax-phase31-control-") as temporary:
            workspace = Path(temporary)
            shutil.copy2(task_root / "environment/starter/kernel.py", workspace / "kernel.py")
            environment = _git_environment()
            subprocess.run(["git", "init", "-q", str(workspace)], check=True, env=environment)
            subprocess.run(["git", "-C", str(workspace), "add", "."], check=True, env=environment)
            subprocess.run(
                ["git", "-C", str(workspace), "commit", "-q", "-m", "task base"],
                check=True,
                env=environment,
            )
            shutil.copy2(source, workspace / "kernel.py")
            patch = subprocess.run(
                ["git", "-C", str(workspace), "diff", "--binary", "HEAD"],
                capture_output=True,
                check=True,
                env=environment,
            ).stdout
        patch_path = out_dir / f"{task_id}.patch"
        patch_path.write_bytes(patch)
        records.append(
            {
                "task_id": task_id,
                "family": family,
                "task_sha256": tasks[task_id]["task_sha256"],
                "source_sha256": file_sha256(source),
                "patch_sha256": file_sha256(patch_path),
            }
        )
    manifest = {
        "schema_version": 1,
        "kind": "opjax_phase31_positive_control_patches",
        "benchmark_release_sha256": release["release_sha256"],
        "records": records,
    }
    manifest["patches_sha256"] = canonical_sha256(manifest)
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return manifest


def assemble(*, release_root: Path, patches_root: Path, results_root: Path, out_path: Path) -> dict[str, Any]:
    release = json.loads((release_root / "manifest.json").read_text())
    patches = json.loads((patches_root / "manifest.json").read_text())
    records = []
    for patch_record in patches["records"]:
        task_id = patch_record["task_id"]
        root = results_root / task_id
        result_path = root / "result.json"
        reward_path = root / "reward.json"
        submission_path = root / "submission.json"
        result = json.loads(result_path.read_text())
        reward = json.loads(reward_path.read_text())
        submission = json.loads(submission_path.read_text())
        accepted = (
            reward.get("reward") == 1
            and result.get("correct") is True
            and result.get("authentic") is True
            and result.get("profiled") is True
            and len(result.get("correctness_cases", ())) == 3
            and submission.get("worker", {}).get("destroyed_at")
        )
        records.append(
            {
                **patch_record,
                "accepted": bool(accepted),
                "reward": reward.get("reward"),
                "performance_eligible": result.get("performance_eligible"),
                "speedup": result.get("speedup"),
                "result_sha256": file_sha256(result_path),
                "reward_sha256": file_sha256(reward_path),
                "submission_sha256": file_sha256(submission_path),
                "worker": submission.get("worker"),
            }
        )
    if {record["task_id"] for record in records} != set(CONTROL_FAMILIES):
        raise G42HarnessError("PHASE31_CONTROL_TASK_SET_INVALID")
    manifest = {
        "schema_version": 1,
        "kind": "opjax_phase31_positive_control_calibration",
        "benchmark_release_sha256": release["release_sha256"],
        "patches_sha256": file_sha256(patches_root / "manifest.json"),
        "families": sorted(set(CONTROL_FAMILIES.values())),
        "accepted": all(record["accepted"] for record in records),
        "records": records,
    }
    manifest["calibration_sha256"] = canonical_sha256(manifest)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="opjax-pallas-phase31-controls")
    sub = parser.add_subparsers(dest="command", required=True)
    patches = sub.add_parser("build-patches")
    patches.add_argument("--release-root", type=Path, required=True)
    patches.add_argument("--control-root", type=Path, required=True)
    patches.add_argument("--out-dir", type=Path, required=True)
    matrix = sub.add_parser("assemble")
    matrix.add_argument("--release-root", type=Path, required=True)
    matrix.add_argument("--patches-root", type=Path, required=True)
    matrix.add_argument("--results-root", type=Path, required=True)
    matrix.add_argument("--out-path", type=Path, required=True)
    args = vars(parser.parse_args(argv))
    command = args.pop("command")
    try:
        result = build_patches(**args) if command == "build-patches" else assemble(**args)
    except (G42HarnessError, OSError, ValueError) as exc:
        print(f"PHASE31_CONTROL_ERROR {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
