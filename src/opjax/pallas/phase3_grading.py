"""Authoritative grading for frozen Phase 3 agent snapshots."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from opjax.pallas.g42_harness import G42HarnessError, canonical_sha256, file_sha256
from opjax.pallas.jaxbench_worker import (
    GcloudDisposableTPUBackend,
    build_request,
    materialize_submission,
    validate_response,
)
from opjax.pallas.phase3_baseline import load_phase3_contract, validate_sample_matrix
from opjax.pallas.phase31_worker import grade_on_gcloud

EMPTY_PATCH_SHA256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"


def normalize_submission_patch(*, source: Path, destination: Path) -> dict[str, Any]:
    raw = source.read_text(encoding="utf-8")
    normalized_lines = []
    changed_lines = 0
    for line in raw.splitlines(keepends=True):
        ending = "\n" if line.endswith("\n") else ""
        body = line[:-1] if ending else line
        if body.startswith("+") and not body.startswith("+++"):
            stripped = body.rstrip(" \t")
            changed_lines += int(stripped != body)
            body = stripped
        normalized_lines.append(body + ending)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("".join(normalized_lines), encoding="utf-8")
    return {
        "kind": "strip_trailing_whitespace_from_added_lines",
        "changed_lines": changed_lines,
        "raw_patch_sha256": file_sha256(source),
        "submission_patch_sha256": file_sha256(destination),
    }


def artifact_failure_record(
    *, record: dict[str, Any], turn: int, patch_path: Path
) -> dict[str, Any]:
    patch_sha256 = file_sha256(patch_path)
    if patch_sha256 != EMPTY_PATCH_SHA256:
        raise G42HarnessError("PHASE3_ARTIFACT_FAILURE_PATCH_NONEMPTY")
    return {
        "unit_id": f"{record['run_id']}--turn-{turn}",
        "model_id": record["model_id"],
        "model_revision": record["model_revision"],
        "provider": record["provider"],
        "task_id": record["task_id"],
        "task_sha256": record["task_sha256"],
        "seed": record["seed"],
        "turn": turn,
        "patch_sha256": patch_sha256,
        "reward": 0,
        "failure_stage": "artifact_contract",
        "candidate_attributable": True,
        "correct": False,
        "authentic": False,
        "profiled": False,
        "speedup": None,
        "beats_xla": False,
        "execution": "trusted_pre_tpu_empty_patch_gate",
        "worker": None,
    }


def patch_contract_failure_record(
    *,
    record: dict[str, Any],
    turn: int,
    patch_path: Path,
    submission_patch_path: Path,
    transformation: dict[str, Any],
    error: str,
) -> dict[str, Any]:
    return {
        "unit_id": f"{record['run_id']}--turn-{turn}",
        "model_id": record["model_id"],
        "model_revision": record["model_revision"],
        "provider": record["provider"],
        "task_id": record["task_id"],
        "task_sha256": record["task_sha256"],
        "seed": record["seed"],
        "turn": turn,
        "patch_sha256": file_sha256(patch_path),
        "submission_patch_sha256": file_sha256(submission_patch_path),
        "patch_transformation": transformation,
        "reward": 0,
        "failure_stage": "artifact_contract",
        "candidate_attributable": True,
        "correct": False,
        "authentic": False,
        "profiled": False,
        "speedup": None,
        "beats_xla": False,
        "execution": "trusted_pre_tpu_patch_gate",
        "error": error,
        "worker": None,
    }


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise G42HarnessError(f"PHASE3_GRADING_JSON_INVALID:{path}") from exc
    if not isinstance(value, dict):
        raise G42HarnessError(f"PHASE3_GRADING_JSON_OBJECT_REQUIRED:{path}")
    return value


def _validate_sample_manifest(
    *, sample_root: Path, experiment: dict[str, Any]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = _read_json(sample_root / "manifest.json")
    payload = dict(manifest)
    expected_hash = payload.pop("release_sha256", None)
    if (
        manifest.get("kind") != "opjax_phase3_sample_matrix"
        or canonical_sha256(payload) != expected_hash
        or manifest.get("experiment_sha256") != experiment["experiment_sha256"]
    ):
        raise G42HarnessError("PHASE3_SAMPLE_MANIFEST_INVALID")
    records = manifest.get("records", [])
    expected = {
        (cell["model_id"], cell["task_id"], cell["seed"])
        for cell in experiment["cells"]
        if cell["provider"] == manifest.get("provider")
    }
    observed = {
        (record.get("model_id"), record.get("task_id"), record.get("seed"))
        for record in records
    }
    if (
        len(records) != manifest.get("counts", {}).get("runs")
        or observed != expected
        or len(observed) != len(records)
    ):
        raise G42HarnessError("PHASE3_SAMPLE_RECORD_COUNT_INVALID")
    for record in records:
        run_root = sample_root / record["run_path"]
        if file_sha256(run_root / "manifest.json") != record.get("run_manifest_sha256"):
            raise G42HarnessError("PHASE3_SAMPLE_RUN_MANIFEST_HASH_INVALID")
        if file_sha256(run_root / "trajectory.json") != record.get("trajectory_sha256"):
            raise G42HarnessError("PHASE3_SAMPLE_TRAJECTORY_HASH_INVALID")
        for turn in (3, 6):
            snapshot = record["snapshots"].get(str(turn), record["snapshots"].get(turn))
            if (
                not isinstance(snapshot, dict)
                or file_sha256(run_root / "snapshots" / f"turn-{turn}.patch")
                != snapshot.get("patch_sha256")
                or file_sha256(run_root / "snapshots" / f"turn-{turn}-kernel.py")
                != snapshot.get("kernel_sha256")
            ):
                raise G42HarnessError("PHASE3_SAMPLE_SNAPSHOT_HASH_INVALID")
    return manifest, records


def _hardware_record(
    *,
    record: dict[str, Any],
    turn: int,
    raw_patch_path: Path,
    submission_patch_path: Path,
    transformation: dict[str, Any],
    response_root: Path,
    request: dict[str, Any],
) -> dict[str, Any]:
    response = validate_response(request=request, destination=response_root)
    reward = _read_json(response_root / "reward.json")
    result = _read_json(response_root / "result.json")
    return {
        "unit_id": f"{record['run_id']}--turn-{turn}",
        "model_id": record["model_id"],
        "model_revision": record["model_revision"],
        "provider": record["provider"],
        "task_id": record["task_id"],
        "task_sha256": record["task_sha256"],
        "seed": record["seed"],
        "turn": turn,
        "patch_sha256": file_sha256(raw_patch_path),
        "submission_patch_sha256": file_sha256(submission_patch_path),
        "patch_transformation": transformation,
        "reward": reward["reward"],
        "failure_stage": reward.get("failure_stage"),
        "candidate_attributable": result.get("candidate_attributable"),
        "correct": reward.get("correct", False),
        "authentic": reward.get("authentic", False),
        "profiled": reward.get("profiled", False),
        "speedup": reward.get("speedup"),
        "beats_xla": reward.get("beats_xla", False),
        "execution": "disposable_tpu_worker",
        "worker": response["worker"],
        "submission_sha256": file_sha256(response_root / "submission.json"),
        "reward_sha256": file_sha256(response_root / "reward.json"),
        "result_sha256": file_sha256(response_root / "result.json"),
    }


def grade_sample_matrix(
    *,
    release_root: Path,
    experiment: dict[str, Any],
    sample_root: Path,
    output_root: Path,
    service_account: str,
    zone: str,
    max_concurrency: int,
    phase31: bool = False,
) -> dict[str, Any]:
    if max_concurrency < 1:
        raise G42HarnessError("PHASE3_GRADING_CONCURRENCY_INVALID")
    release = _read_json(release_root / "manifest.json")
    _, sample_records = _validate_sample_manifest(
        sample_root=sample_root,
        experiment=experiment,
    )
    release_tasks = {task["task_id"]: task for task in release["tasks"]}
    output_root.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    hardware_units: list[
        tuple[dict[str, Any], int, Path, Path, dict[str, Any], Path]
    ] = []
    for sample in sample_records:
        for turn in (3, 6):
            patch_path = sample_root / sample["run_path"] / "snapshots" / f"turn-{turn}.patch"
            if file_sha256(patch_path) == EMPTY_PATCH_SHA256:
                records.append(
                    artifact_failure_record(
                        record=sample,
                        turn=turn,
                        patch_path=patch_path,
                    )
                )
            else:
                unit_root = output_root / "results" / f"{sample['run_id']}--turn-{turn}"
                submission_patch = (
                    output_root
                    / "normalized-patches"
                    / f"{sample['run_id']}--turn-{turn}.patch"
                )
                transformation = normalize_submission_patch(
                    source=patch_path,
                    destination=submission_patch,
                )
                task = release_tasks[sample["task_id"]]
                try:
                    with tempfile.TemporaryDirectory(
                        prefix="opjax-phase3-patch-preflight-"
                    ) as temporary:
                        materialize_submission(
                            task_root=release_root / task["path"],
                            patch_path=submission_patch,
                            destination=Path(temporary) / "workspace",
                        )
                except Exception as exc:
                    records.append(
                        patch_contract_failure_record(
                            record=sample,
                            turn=turn,
                            patch_path=patch_path,
                            submission_patch_path=submission_patch,
                            transformation=transformation,
                            error=f"{type(exc).__name__}: {exc}",
                        )
                    )
                else:
                    hardware_units.append(
                        (
                            sample,
                            turn,
                            patch_path,
                            submission_patch,
                            transformation,
                            unit_root,
                        )
                    )

    def grade_hardware(
        unit: tuple[dict[str, Any], int, Path, Path, dict[str, Any], Path]
    ) -> dict[str, Any]:
        (
            sample,
            turn,
            raw_patch_path,
            submission_patch_path,
            transformation,
            unit_root,
        ) = unit
        task = release_tasks[sample["task_id"]]
        request = build_request(
            release=release,
            task=task,
            patch_path=submission_patch_path,
        )
        response_root = unit_root / "artifacts"
        if response_root.exists():
            return _hardware_record(
                record=sample,
                turn=turn,
                raw_patch_path=raw_patch_path,
                submission_patch_path=submission_patch_path,
                transformation=transformation,
                response_root=response_root,
                request=request,
            )
        unit_root.mkdir(parents=True, exist_ok=False)
        if phase31:
            grade_on_gcloud(
                release_root=release_root,
                request=request,
                patch_path=submission_patch_path,
                destination=response_root,
                service_account=service_account,
                zone=zone,
            )
        else:
            backend = GcloudDisposableTPUBackend(
                release_root=release_root,
                patch_path=submission_patch_path,
                service_account=service_account,
                zone=zone,
                name_prefix="opjax-p3",
            )
            backend.grade(request, response_root)
        return _hardware_record(
            record=sample,
            turn=turn,
            raw_patch_path=raw_patch_path,
            submission_patch_path=submission_patch_path,
            transformation=transformation,
            response_root=response_root,
            request=request,
        )

    with ThreadPoolExecutor(max_workers=max_concurrency) as executor:
        futures = {
            executor.submit(grade_hardware, unit): unit[0]["run_id"]
            for unit in hardware_units
        }
        completed = 0
        for future in as_completed(futures):
            unit_id = futures[future]
            try:
                records.append(future.result())
            except Exception as exc:
                for pending in futures:
                    pending.cancel()
                print(
                    f"PHASE3_HARDWARE_GRADE_FAILED unit={unit_id} "
                    f"error={type(exc).__name__}:{exc}",
                    file=sys.stderr,
                )
                raise G42HarnessError(
                    f"PHASE3_HARDWARE_GRADE_FAILED:{unit_id}"
                ) from exc
            completed += 1
            print(
                f"PHASE3_HARDWARE_GRADE completed={completed}/"
                f"{len(hardware_units)} unit={unit_id}",
                file=sys.stderr,
            )
    records.sort(key=lambda record: record["unit_id"])
    horizons = {}
    for turn in (3, 6):
        subset = [record for record in records if record["turn"] == turn]
        horizons[f"k{turn}"] = {
            "units": len(subset),
            "profile_verified": sum(record["reward"] == 1 for record in subset),
            "candidate_failures": sum(record["reward"] == 0 for record in subset),
            "infrastructure_failures": sum(record["reward"] == -1 for record in subset),
            "nonempty_patches": sum(
                record["patch_sha256"] != EMPTY_PATCH_SHA256 for record in subset
            ),
            "beats_xla": sum(record["beats_xla"] is True for record in subset),
        }
    result = {
        "schema_version": 2 if phase31 else 1,
        "kind": (
            "opjax_phase31_base_capability_result"
            if phase31
            else "opjax_phase3_base_capability_result"
        ),
        "experiment_sha256": experiment["experiment_sha256"],
        "benchmark_release_sha256": release["release_sha256"],
        "sample_release_sha256": _read_json(sample_root / "manifest.json")[
            "release_sha256"
        ],
        "provider": _read_json(sample_root / "manifest.json")["provider"],
        "counts": {
            "trajectories": len(sample_records),
            "snapshots": len(records),
            "hardware_graded": len(hardware_units),
            "artifact_gate_failures": len(records) - len(hardware_units),
        },
        "horizons": horizons,
        "records": records,
    }
    result["result_sha256"] = canonical_sha256(result)
    path = output_root / "result.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="opjax-pallas-phase3-grade")
    parser.add_argument(
        "--release-root",
        type=Path,
        default=Path("data/pallas/benchmarks/jaxbench-v1"),
    )
    parser.add_argument(
        "--closeout-root",
        type=Path,
        default=Path("data/pallas/runs/jaxbench-full-v1-closeout"),
    )
    parser.add_argument(
        "--experiment",
        type=Path,
        default=Path("data/pallas/runs/phase3-base-capability/experiment.json"),
    )
    parser.add_argument("--sample-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--service-account", required=True)
    parser.add_argument("--zone", default="us-west4-a")
    parser.add_argument("--max-concurrency", type=int, default=4)
    args = parser.parse_args(argv)
    try:
        contract = load_phase3_contract(
            release_root=args.release_root,
            closeout_root=args.closeout_root,
        )
        validate_sample_matrix(path=args.experiment, contract=contract)
        experiment = _read_json(args.experiment)
        result = grade_sample_matrix(
            release_root=args.release_root.resolve(),
            experiment=experiment,
            sample_root=args.sample_root.resolve(),
            output_root=args.output_root.resolve(),
            service_account=args.service_account,
            zone=args.zone,
            max_concurrency=args.max_concurrency,
        )
    except (G42HarnessError, OSError, ValueError) as exc:
        print(f"PHASE3_GRADING_ERROR {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
