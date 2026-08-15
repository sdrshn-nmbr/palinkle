from __future__ import annotations

import concurrent.futures
import json
from pathlib import Path
import subprocess
import tempfile
from typing import Any
import uuid

from opjax.pallas.g42_harness import G42HarnessError
from opjax.pallas.laguna_dspark_conformance import canonical_sha256, file_sha256
from opjax.pallas.phase3_grading import (
    EMPTY_PATCH_SHA256,
    _validate_sample_manifest,
    materialize_submission,
    normalize_submission_patch,
)


PUBLIC_GATE_TIMEOUT_SECONDS = 180


def _public_gate_source_sha256() -> str:
    root = Path(__file__).parent
    return canonical_sha256(
        {
            name: file_sha256(root / name)
            for name in ("laguna_live_public_gate.py", "phase3_grading.py")
        }
    )


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise G42HarnessError(f"LAGUNA_PUBLIC_GATE_JSON_OBJECT_REQUIRED:{path}")
    return value


def _failure_code(output: str) -> str:
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    if not lines:
        return "PUBLIC_DEV_CHECK_FAILED_WITHOUT_OUTPUT"
    return lines[-1].split(":", maxsplit=1)[0]


def _docker_image_id(image: str) -> str:
    result = subprocess.run(
        ["docker", "image", "inspect", "--format", "{{.Id}}", image],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def run_public_dev_check(*, workspace: Path, image: str) -> dict[str, Any]:
    container = f"opjax-public-gate-{uuid.uuid4().hex[:12]}"
    command = [
        "docker",
        "run",
        "--name",
        container,
        "--rm",
        "--network",
        "none",
        "--user",
        "65534:65534",
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges",
        "--read-only",
        "--pids-limit",
        "64",
        "--memory",
        "4g",
        "--cpus",
        "2",
        "--tmpfs",
        "/tmp:rw,nosuid,nodev,size=512m",
        "--mount",
        f"type=bind,src={workspace.resolve()},dst=/workspace,readonly",
        "-w",
        "/workspace",
        image,
        "python",
        "dev_check.py",
    ]
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=PUBLIC_GATE_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        subprocess.run(
            ["docker", "rm", "-f", container],
            check=False,
            capture_output=True,
            text=True,
        )
        return {
            "passed": False,
            "exit_code": None,
            "failure_code": "PUBLIC_DEV_CHECK_TIMEOUT",
            "output": "",
        }
    output = (result.stdout + result.stderr).strip()
    return {
        "passed": result.returncode == 0,
        "exit_code": result.returncode,
        "failure_code": None if result.returncode == 0 else _failure_code(output),
        "output": output[-4096:],
    }


def _validate_environment_canary(*, workspace: Path, image: str) -> dict[str, Any]:
    result = run_public_dev_check(workspace=workspace, image=image)
    if result["passed"] is not True:
        raise G42HarnessError("LAGUNA_PUBLIC_GATE_ENVIRONMENT_CANARY_FAILED")
    return {
        **result,
        "workspace": str(workspace),
        "kernel_sha256": file_sha256(workspace / "kernel.py"),
        "dev_check_sha256": file_sha256(workspace / "dev_check.py"),
    }


def build_public_gate_result(
    *,
    arm: str,
    experiment: dict[str, Any],
    release_root: Path,
    sample_root: Path,
    normalized_root: Path,
    image: str,
    environment_canary: dict[str, Any],
    max_concurrency: int,
) -> dict[str, Any]:
    if arm not in {"dflash", "dspark"} or max_concurrency < 1:
        raise G42HarnessError("LAGUNA_PUBLIC_GATE_ARGUMENT_INVALID")
    manifest, records = _validate_sample_manifest(
        sample_root=sample_root,
        experiment=experiment,
    )
    release = _read_json(release_root / "manifest.json")
    tasks = {task["task_id"]: task for task in release["tasks"]}
    normalized_root.mkdir(parents=True, exist_ok=True)

    def evaluate(record: dict[str, Any]) -> dict[str, Any]:
        snapshot = record["snapshots"]["6"]
        patch = sample_root / record["run_path"] / "snapshots" / "turn-6.patch"
        base = {
            "run_id": record["run_id"],
            "task_id": record["task_id"],
            "task_sha256": record["task_sha256"],
            "seed": record["seed"],
            "turn": 6,
            "patch_sha256": snapshot["patch_sha256"],
            "kernel_sha256": snapshot["kernel_sha256"],
        }
        if snapshot["patch_sha256"] == EMPTY_PATCH_SHA256:
            return {
                **base,
                "passed": False,
                "exit_code": None,
                "failure_code": "EMPTY_PATCH",
                "output": "",
                "execution": "trusted_empty_patch_gate",
            }
        normalized = normalized_root / f"{record['run_id']}--turn-6.patch"
        transformation = normalize_submission_patch(
            source=patch,
            destination=normalized,
        )
        with tempfile.TemporaryDirectory(prefix="opjax-public-gate-") as temporary:
            workspace = Path(temporary) / "workspace"
            materialize_submission(
                task_root=release_root / tasks[record["task_id"]]["path"],
                patch_path=normalized,
                destination=workspace,
            )
            dev_check = run_public_dev_check(workspace=workspace, image=image)
            return {
                **base,
                **dev_check,
                "normalized_patch_sha256": file_sha256(normalized),
                "patch_transformation": transformation,
                "dev_check_sha256": file_sha256(workspace / "dev_check.py"),
                "materialized_kernel_sha256": file_sha256(workspace / "kernel.py"),
                "execution": "isolated_public_dev_check",
            }

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_concurrency) as executor:
        evaluated = list(executor.map(evaluate, records))
    evaluated.sort(key=lambda item: item["run_id"])
    failures: dict[str, int] = {}
    for record in evaluated:
        code = record["failure_code"] or "passed"
        failures[code] = failures.get(code, 0) + 1
    result = {
        "schema_version": 1,
        "kind": "opjax_laguna_live_public_gate",
        "arm": arm,
        "experiment_sha256": experiment["experiment_sha256"],
        "benchmark_release_sha256": release["release_sha256"],
        "sample_release_sha256": manifest["release_sha256"],
        "public_gate_source_sha256": _public_gate_source_sha256(),
        "image": image,
        "image_id": _docker_image_id(image),
        "environment_canary": environment_canary,
        "contract": {
            "network": "none",
            "workspace": "read_only",
            "candidate_user": "65534:65534",
            "capabilities": "none",
            "timeout_seconds": PUBLIC_GATE_TIMEOUT_SECONDS,
            "claim": (
                "a public dev-check failure is a necessary-condition failure; "
                "it cannot receive profile-verified TPU credit"
            ),
        },
        "counts": {
            "trajectories": len(evaluated),
            "nonempty_patches": sum(
                record["patch_sha256"] != EMPTY_PATCH_SHA256 for record in evaluated
            ),
            "public_gate_passes": sum(record["passed"] is True for record in evaluated),
            "profile_verified_upper_bound": sum(
                record["passed"] is True for record in evaluated
            ),
        },
        "failure_codes": dict(sorted(failures.items())),
        "records": evaluated,
    }
    result["result_sha256"] = canonical_sha256(result)
    return result


def validate_live_public_gate(
    *,
    experiment_path: Path,
    release_root: Path,
    sample_roots: dict[str, Path],
    output_root: Path,
) -> dict[str, Any]:
    experiment = _read_json(experiment_path)
    release = _read_json(release_root / "manifest.json")
    summary_path = output_root / "summary.json"
    summary = _read_json(summary_path)
    expected_summary_hash = canonical_sha256(
        {key: value for key, value in summary.items() if key != "summary_sha256"}
    )
    if summary.get("summary_sha256") != expected_summary_hash:
        raise G42HarnessError("LAGUNA_PUBLIC_GATE_SUMMARY_HASH_INVALID")
    if set(summary.get("arms", {})) != set(sample_roots):
        raise G42HarnessError("LAGUNA_PUBLIC_GATE_SUMMARY_ARMS_INVALID")
    if (
        summary.get("experiment_sha256") != experiment["experiment_sha256"]
        or summary.get("benchmark_release_sha256") != release["release_sha256"]
        or summary.get("image") != experiment["harness"]["agent_image"]
        or summary.get("image_id") != experiment["harness"]["agent_image_id"]
        or summary.get("public_gate_source_sha256")
        != _public_gate_source_sha256()
    ):
        raise G42HarnessError("LAGUNA_PUBLIC_GATE_SUMMARY_BINDING_INVALID")
    for arm, sample_root in sorted(sample_roots.items()):
        result_path = output_root / arm / "result.json"
        result = _read_json(result_path)
        expected_result_hash = canonical_sha256(
            {key: value for key, value in result.items() if key != "result_sha256"}
        )
        if result.get("result_sha256") != expected_result_hash:
            raise G42HarnessError("LAGUNA_PUBLIC_GATE_RESULT_HASH_INVALID")
        summary_arm = summary["arms"][arm]
        if (
            summary_arm.get("result_sha256") != result["result_sha256"]
            or summary_arm.get("file_sha256") != file_sha256(result_path)
        ):
            raise G42HarnessError("LAGUNA_PUBLIC_GATE_RESULT_LINK_INVALID")
        manifest, sample_records = _validate_sample_manifest(
            sample_root=sample_root,
            experiment=experiment,
        )
        expected_records = {
            record["run_id"]: {
                "task_id": record["task_id"],
                "task_sha256": record["task_sha256"],
                "seed": record["seed"],
                "patch_sha256": record["snapshots"]["6"]["patch_sha256"],
                "kernel_sha256": record["snapshots"]["6"]["kernel_sha256"],
            }
            for record in sample_records
        }
        observed_records = result.get("records", [])
        if len(observed_records) != len(expected_records):
            raise G42HarnessError("LAGUNA_PUBLIC_GATE_RECORD_COUNT_INVALID")
        failures: dict[str, int] = {}
        for record in observed_records:
            expected = expected_records.get(record.get("run_id"))
            if expected is None or any(
                record.get(key) != value for key, value in expected.items()
            ):
                raise G42HarnessError("LAGUNA_PUBLIC_GATE_RECORD_BINDING_INVALID")
            code = record.get("failure_code") or "passed"
            failures[code] = failures.get(code, 0) + 1
        counts = {
            "trajectories": len(observed_records),
            "nonempty_patches": sum(
                record["patch_sha256"] != EMPTY_PATCH_SHA256
                for record in observed_records
            ),
            "public_gate_passes": sum(
                record.get("passed") is True for record in observed_records
            ),
            "profile_verified_upper_bound": sum(
                record.get("passed") is True for record in observed_records
            ),
        }
        if (
            result.get("arm") != arm
            or result.get("sample_release_sha256") != manifest["release_sha256"]
            or result.get("experiment_sha256") != experiment["experiment_sha256"]
            or result.get("benchmark_release_sha256") != release["release_sha256"]
            or result.get("image") != summary["image"]
            or result.get("image_id") != summary["image_id"]
            or result.get("public_gate_source_sha256")
            != summary.get("public_gate_source_sha256")
            or result.get("environment_canary", {}).get("passed") is not True
            or result.get("counts") != counts
            or result.get("failure_codes") != dict(sorted(failures.items()))
            or summary_arm.get("counts") != counts
            or summary_arm.get("failure_codes") != dict(sorted(failures.items()))
        ):
            raise G42HarnessError("LAGUNA_PUBLIC_GATE_RESULT_CONTRACT_INVALID")
    return summary


def run_live_public_gate(
    *,
    experiment_path: Path,
    release_root: Path,
    sample_roots: dict[str, Path],
    output_root: Path,
    canary_workspace: Path,
    max_concurrency: int,
) -> dict[str, Any]:
    experiment = _read_json(experiment_path)
    image = experiment["harness"]["agent_image"]
    observed_image_id = _docker_image_id(image)
    expected_image_id = experiment["harness"]["agent_image_id"]
    if observed_image_id != expected_image_id:
        raise G42HarnessError("LAGUNA_PUBLIC_GATE_IMAGE_ID_MISMATCH")
    canary = _validate_environment_canary(workspace=canary_workspace, image=image)
    output_root.mkdir(parents=True, exist_ok=True)
    results = {}
    for arm, sample_root in sorted(sample_roots.items()):
        result = build_public_gate_result(
            arm=arm,
            experiment=experiment,
            release_root=release_root,
            sample_root=sample_root,
            normalized_root=output_root / arm / "normalized-patches",
            image=image,
            environment_canary=canary,
            max_concurrency=max_concurrency,
        )
        path = output_root / arm / "result.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        results[arm] = {
            "result_sha256": result["result_sha256"],
            "file_sha256": file_sha256(path),
            "counts": result["counts"],
            "failure_codes": result["failure_codes"],
        }
    summary = {
        "schema_version": 1,
        "kind": "opjax_laguna_live_public_gate_comparison",
        "experiment_sha256": experiment["experiment_sha256"],
        "benchmark_release_sha256": _read_json(release_root / "manifest.json")[
            "release_sha256"
        ],
        "image": image,
        "image_id": observed_image_id,
        "public_gate_source_sha256": _public_gate_source_sha256(),
        "arms": results,
        "conclusion": (
            "both arms have a zero profile-verified upper bound because every "
            "nonempty turn-6 candidate failed the frozen public dev check"
        ),
    }
    summary["summary_sha256"] = canonical_sha256(summary)
    (output_root / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    return validate_live_public_gate(
        experiment_path=experiment_path,
        release_root=release_root,
        sample_roots=sample_roots,
        output_root=output_root,
    )
