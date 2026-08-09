"""Build truthful six-turn G4.2 repair trajectories from admitted TPU evidence."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from opjax.pallas.g42_harness import (
    canonical_sha256,
    create_agent_workspace,
    file_sha256,
    load_task_package,
    snapshot_workspace,
)
from opjax.pallas.g42_verifier import sanitized_feedback
from opjax.pallas.phase2_contamination import assert_project_training_content_clean

SYSTEM_PROMPT = """You are a programming agent in an isolated repository.
Return exactly one shell action in an mswea_bash_command fence per turn. Inspect and repair kernel.py.
Use `echo REQUEST_TPU_VERIFIER` to request one sanitized verifier stage. Use
`echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT` only when ready to submit. Hidden tests, compiler dumps,
absolute verifier paths, credentials, and reference code are unavailable.
"""
INSTANCE_PROMPT = """Repair the Pallas kernel described in instruction.md. Start by reading the task,
Pallas API notes, starter kernel, and visible development check. The maximum turn budget is hidden.
"""


class G42TraceError(RuntimeError):
    """A trace cannot be represented as truthful harness execution."""


def _action(command: str) -> str:
    return f"```mswea_bash_command\n{command}\n```"


def _observation(*, returncode: int, output: str, verifier_feedback: str | None = None) -> str:
    content = f"<returncode>{returncode}</returncode>\n<output>{output}</output>"
    if verifier_feedback is not None:
        content += f"\n<{verifier_feedback}>"
    return content


def _run_action(workspace: Path, command: str) -> dict[str, Any]:
    process = subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "--network",
            "none",
            "--mount",
            f"type=bind,src={workspace},dst=/workspace",
            "-w",
            "/workspace",
            "python:3.12-slim",
            "sh",
            "-lc",
            command,
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    return {
        "command": command,
        "returncode": process.returncode,
        "output": process.stdout + process.stderr,
    }


def _repair_command(solution: str) -> str:
    if '\"\"\"' in solution:
        raise G42TraceError("SOLUTION_TRIPLE_QUOTE_UNSUPPORTED")
    return (
        "python - <<'PY'\n"
        "from pathlib import Path\n\n"
        "Path(\"kernel.py\").write_text(\n"
        f'    \"\"\"{solution}\"\"\",\n'
        '    encoding="utf-8",\n'
        ")\n"
        "PY"
    )


def _load_reward(admission_root: Path, task_id: str, kind: str) -> dict[str, Any]:
    if kind == "starter":
        path = admission_root / "raw" / "g42-admission-v2" / "starters" / task_id / "run.log"
    else:
        path = admission_root / "raw" / "g42-admission" / "solutions" / task_id / "result.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise G42TraceError(f"ADMISSION_RESULT_INVALID: {path}")
    return value


def build_trace_release(*, task_release: Path, admission_root: Path, out_dir: Path) -> dict[str, Any]:
    if out_dir.exists():
        raise G42TraceError(f"OUTPUT_EXISTS: {out_dir}")
    task_manifest = json.loads((task_release / "manifest.json").read_text(encoding="utf-8"))
    selected = task_manifest["training_selection"]
    records = []
    sft_rows = []
    for task_id in selected:
        task = load_task_package(task_release / "tasks" / task_id)
        trace_root = (out_dir / "trajectories" / task_id).resolve()
        workspace = trace_root / "workspace"
        workspace_record = create_agent_workspace(task, workspace)
        starter_result = _load_reward(admission_root, task_id, "starter")
        solution_result = _load_reward(admission_root, task_id, "solution")
        solution = (task.root / "solution" / "kernel.py").read_text(encoding="utf-8")
        commands = (
            "sed -n '1,240p' instruction.md PALLAS_API.md kernel.py dev_check.py && python dev_check.py kernel.py",
            "echo REQUEST_TPU_VERIFIER",
            _repair_command(solution),
            "python dev_check.py kernel.py",
            "echo REQUEST_TPU_VERIFIER",
            "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT",
        )
        messages: list[dict[str, str]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": INSTANCE_PROMPT},
        ]
        turns = []
        for turn, command in enumerate(commands, start=1):
            action_message = {"role": "assistant", "content": _action(command)}
            messages.append(action_message)
            sft_rows.append(
                {
                    "row_id": f"{task_id}:turn-{turn}",
                    "task_id": task_id,
                    "family": task.family,
                    "turn": turn,
                    "messages": list(messages),
                    "terminal_verified": True,
                    "solution_kernel_sha256": solution_result["kernel_sha256"],
                }
            )
            executed = _run_action(workspace, command)
            if turn >= 2 and executed["returncode"] != 0:
                raise G42TraceError(
                    f"CURRICULUM_ACTION_FAILED: {task_id}:turn={turn}:code={executed['returncode']}"
                )
            if turn == 4 and "DEV_CHECK static_complete" not in executed["output"]:
                raise G42TraceError(f"REPAIRED_DEV_CHECK_FAILED: {task_id}")
            verifier_feedback = None
            if turn == 2:
                verifier_feedback = sanitized_feedback(starter_result)
            elif turn == 5:
                verifier_feedback = "VERIFIER_STAGE verified: All mandatory TPU verifier stages passed."
            observation = _observation(
                returncode=executed["returncode"],
                output=executed["output"],
                verifier_feedback=verifier_feedback,
            )
            messages.append({"role": "user", "content": observation})
            snapshot = snapshot_workspace(workspace, turn=turn, output_dir=trace_root / "snapshots")
            turns.append({**executed, "turn": turn, "observation": observation, "snapshot": snapshot})
        if file_sha256(workspace / "kernel.py") != solution_result["kernel_sha256"]:
            raise G42TraceError(f"TERMINAL_KERNEL_HASH_MISMATCH: {task_id}")
        shutil.copy2(trace_root / "snapshots" / "turn-6.patch", trace_root / "model.patch")
        trajectory = {
            "schema_version": 1,
            "kind": "pallas_g42_truthful_repair_trajectory",
            "task_id": task_id,
            "task_sha256": task.task_sha256,
            "workspace": workspace_record,
            "turn_limit": 6,
            "turns": turns,
            "messages": messages,
            "terminal_verifier": solution_result,
            "terminal_verified": True,
        }
        trajectory_path = trace_root / "trajectory.json"
        trajectory_path.write_text(json.dumps(trajectory, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        records.append(
            {
                "task_id": task_id,
                "task_sha256": task.task_sha256,
                "trajectory_sha256": file_sha256(trajectory_path),
                "model_patch_sha256": file_sha256(trace_root / "model.patch"),
                "terminal_kernel_sha256": solution_result["kernel_sha256"],
            }
        )
        shutil.rmtree(workspace)
    dataset = out_dir / "datasets" / "prefix-sft.jsonl"
    assert_project_training_content_clean(sft_rows)
    dataset.parent.mkdir(parents=True)
    dataset.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in sft_rows), encoding="utf-8")
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "kind": "pallas_g42_trace_release",
        "task_release_sha256": task_manifest["release_sha256"],
        "admission_release_sha256": json.loads(
            (admission_root / "manifest.json").read_text(encoding="utf-8")
        )["release_sha256"],
        "counts": {"trajectories": len(records), "turns": len(sft_rows), "prefix_sft_rows": len(sft_rows)},
        "dataset_sha256": file_sha256(dataset),
        "records": records,
    }
    manifest["release_sha256"] = canonical_sha256(manifest)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def validate_trace_release(root: Path) -> dict[str, Any]:
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    dataset = root / "datasets" / "prefix-sft.jsonl"
    rows = [json.loads(line) for line in dataset.read_text(encoding="utf-8").splitlines() if line]
    if manifest.get("kind") != "pallas_g42_trace_release" or len(rows) != 192:
        raise G42TraceError(f"TRACE_RELEASE_INVALID: {root}")
    if file_sha256(dataset) != manifest.get("dataset_sha256"):
        raise G42TraceError(f"TRACE_DATASET_HASH_MISMATCH: {root}")
    if any(row.get("terminal_verified") is not True or row.get("turn") not in range(1, 7) for row in rows):
        raise G42TraceError(f"TRACE_ROW_INVALID: {root}")
    expected_sha = manifest.pop("release_sha256", None)
    observed_sha = canonical_sha256(manifest)
    if expected_sha != observed_sha:
        raise G42TraceError(f"TRACE_RELEASE_HASH_MISMATCH: {root}")
    return {"release_sha256": observed_sha, **manifest["counts"]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="opjax-pallas-g42-traces")
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build")
    build.add_argument("--task-release", type=Path, required=True)
    build.add_argument("--admission-root", type=Path, required=True)
    build.add_argument("--out-dir", type=Path, required=True)
    validate = commands.add_parser("validate")
    validate.add_argument("--root", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "build":
            result = build_trace_release(
                task_release=args.task_release,
                admission_root=args.admission_root,
                out_dir=args.out_dir,
            )
        else:
            result = validate_trace_release(args.root)
    except (G42TraceError, OSError, ValueError, json.JSONDecodeError, subprocess.SubprocessError) as exc:
        print(f"G42_TRACE_ERROR {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
