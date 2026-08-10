"""Build and validate the complete Phase 3.1 JAXBench task release."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

from opjax.pallas.agent_protocol import SHELL_TOOLS
from opjax.pallas.g42_harness import canonical_sha256, file_sha256
from opjax.pallas.jaxbench_capability import (
    _contamination_signatures,
    _shingles,
    _task_hash,
    _task_toml,
    validate_release as validate_source_release,
)
from opjax.pallas.phase31_oracle import oracle_contract
from opjax.pallas.phase31_public import PALLAS_API, render_dev_check
from opjax.pallas.phase31_controls import CONTROL_FAMILIES


class Phase31BenchmarkError(RuntimeError):
    pass


BOUND_SOURCES = (
    "benchmarking.py",
    "agent_protocol.py",
    "jaxbench_executable.py",
    "jaxbench_verifier.py",
    "jaxbench_worker.py",
    "phase31_benchmark.py",
    "phase31_controls.py",
    "phase31_oracle.py",
    "phase31_public.py",
    "phase31_validity.py",
    "phase31_verifier.py",
    "phase31_worker.py",
)


def _write(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


def build_release(
    *,
    source_release: Path,
    source_checkout: Path,
    out_dir: Path,
    agent_image: str,
    agent_image_id: str,
) -> dict[str, Any]:
    if out_dir.exists():
        raise Phase31BenchmarkError(f"OUTPUT_EXISTS:{out_dir}")
    validate_source_release(root=source_release, source_root=source_checkout)
    shutil.copytree(source_release, out_dir)
    source_manifest = json.loads((source_release / "manifest.json").read_text())
    task_records = []
    for source_record in source_manifest["tasks"]:
        task_root = out_dir / source_record["path"]
        task_json_path = task_root / "tests/task.json"
        task = json.loads(task_json_path.read_text())
        task["oracle_contract"] = oracle_contract(
            task["input_argument_names"], task["tensor_schema"]["outputs"][0]["dtype"]
        )
        _write(task_json_path, json.dumps(task, indent=2, sort_keys=True) + "\n")
        _write(task_root / "environment/public/PALLAS_API.md", PALLAS_API)
        _write(
            task_root / "environment/public/dev_check.py",
            render_dev_check(task["tensor_schema"]),
        )
        _write(
            task_root / "task.toml",
            _task_toml(
                workload_id=task["task_id"],
                task_sha256="",
                baseline_sha256=task["baseline_sha256"],
                optimized_sha256=task.get("optimized_sha256"),
            ),
        )
        task_sha = _task_hash(task_root)
        _write(
            task_root / "task.toml",
            _task_toml(
                workload_id=task["task_id"],
                task_sha256=task_sha,
                baseline_sha256=task["baseline_sha256"],
                optimized_sha256=task.get("optimized_sha256"),
            ),
        )
        task_records.append(
            {
                **source_record,
                "task_sha256": task_sha,
                "phase3_task_sha256": source_record["task_sha256"],
                "public_specification_sha256": file_sha256(task_root / "instruction.md"),
                "public_api_sha256": file_sha256(
                    task_root / "environment/public/PALLAS_API.md"
                ),
                "public_dev_check_sha256": file_sha256(
                    task_root / "environment/public/dev_check.py"
                ),
                "oracle_contract_sha256": canonical_sha256(task["oracle_contract"]),
            }
        )
    root = Path(__file__).parent
    repo_root = Path(__file__).parents[3]
    signatures_path = out_dir / "contamination-signatures.json"
    signatures = _contamination_signatures(out_dir)
    control_root = repo_root / "config/pallas/phase31-controls"
    for task_id in CONTROL_FAMILIES:
        source = control_root / f"{task_id}.py"
        text = source.read_text(encoding="utf-8")
        signatures["documents"].append(
            {
                "task_id": task_id,
                "path": f"phase31-controls/{task_id}.py",
                "sha256": file_sha256(source),
                "shingles": _shingles(text),
            }
        )
    _write(
        signatures_path,
        json.dumps(signatures, indent=2, sort_keys=True) + "\n",
    )
    agent_lock = repo_root / "config/pallas/phase31-agent-requirements.lock"
    worker_lock = repo_root / "config/pallas/phase2-worker-requirements.lock"
    manifest = {
        "schema_version": 1,
        "kind": "opjax_phase31_jaxbench_benchmark",
        "benchmark_id": "opjax-jaxbench-phase31",
        "status": "frozen",
        "source_release_sha256": source_manifest["release_sha256"],
        "jaxbench_revision": source_manifest["jaxbench_revision"],
        "shape_policy": "original_unmodified",
        "task_count": len(task_records),
        "action_protocol": {
            "native_tools": sorted(SHELL_TOOLS | {"read", "write", "edit", "list", "ls"}),
            "source_sha256": file_sha256(root / "agent_protocol.py"),
        },
        "agent_environment": {
            "image": agent_image,
            "image_id": agent_image_id,
            "requirements_lock_sha256": file_sha256(agent_lock),
            "dockerfile_sha256": file_sha256(
                repo_root / "config/pallas/phase31-agent.Dockerfile"
            ),
        },
        "worker_requirements_lock_sha256": file_sha256(worker_lock),
        "positive_control_source_sha256": {
            task_id: file_sha256(control_root / f"{task_id}.py")
            for task_id in CONTROL_FAMILIES
        },
        "bound_source_sha256": {
            name: file_sha256(root / name) for name in BOUND_SOURCES
        },
        "contamination_signatures_sha256": file_sha256(signatures_path),
        "tasks": task_records,
    }
    manifest["release_sha256"] = canonical_sha256(manifest)
    _write(out_dir / "manifest.json", json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    validate_release(root=out_dir, source_release=source_release)
    return manifest


def validate_release(*, root: Path, source_release: Path) -> dict[str, Any]:
    manifest = json.loads((root / "manifest.json").read_text())
    source_manifest = json.loads((source_release / "manifest.json").read_text())
    payload = dict(manifest)
    release_sha = payload.pop("release_sha256", None)
    signatures = root / "contamination-signatures.json"
    source_root = Path(__file__).parent
    repo_root = Path(__file__).parents[3]
    control_root = repo_root / "config/pallas/phase31-controls"
    if (
        manifest.get("kind") != "opjax_phase31_jaxbench_benchmark"
        or canonical_sha256(payload) != release_sha
        or manifest.get("source_release_sha256") != source_manifest.get("release_sha256")
        or manifest.get("task_count") != 50
        or manifest.get("shape_policy") != "original_unmodified"
        or manifest.get("bound_source_sha256")
        != {name: file_sha256(source_root / name) for name in BOUND_SOURCES}
        or manifest.get("action_protocol")
        != {
            "native_tools": sorted(SHELL_TOOLS | {"read", "write", "edit", "list", "ls"}),
            "source_sha256": file_sha256(source_root / "agent_protocol.py"),
        }
        or manifest.get("positive_control_source_sha256")
        != {
            task_id: file_sha256(control_root / f"{task_id}.py")
            for task_id in CONTROL_FAMILIES
        }
        or manifest.get("agent_environment", {}).get("requirements_lock_sha256")
        != file_sha256(repo_root / "config/pallas/phase31-agent-requirements.lock")
        or manifest.get("agent_environment", {}).get("dockerfile_sha256")
        != file_sha256(repo_root / "config/pallas/phase31-agent.Dockerfile")
        or manifest.get("contamination_signatures_sha256") != file_sha256(signatures)
    ):
        raise Phase31BenchmarkError("PHASE31_RELEASE_MANIFEST_INVALID")
    source_tasks = {task["task_id"]: task for task in source_manifest["tasks"]}
    if {task["task_id"] for task in manifest["tasks"]} != set(source_tasks):
        raise Phase31BenchmarkError("PHASE31_TASK_SET_INVALID")
    for record in manifest["tasks"]:
        task_root = root / record["path"]
        task = json.loads((task_root / "tests/task.json").read_text())
        expected_oracle = oracle_contract(
            task["input_argument_names"], task["tensor_schema"]["outputs"][0]["dtype"]
        )
        if (
            record["phase3_task_sha256"] != source_tasks[record["task_id"]]["task_sha256"]
            or record["task_sha256"] != _task_hash(task_root)
            or task.get("oracle_contract") != expected_oracle
            or record["oracle_contract_sha256"] != canonical_sha256(expected_oracle)
            or (task_root / "environment/public/PALLAS_API.md").read_text() != PALLAS_API
            or (task_root / "environment/public/dev_check.py").read_text()
            != render_dev_check(task["tensor_schema"])
        ):
            raise Phase31BenchmarkError(f"PHASE31_TASK_INVALID:{record['task_id']}")
    return {"release_sha256": release_sha, "task_count": 50}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="opjax-pallas-phase31-benchmark")
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build")
    build.add_argument("--source-release", type=Path, required=True)
    build.add_argument("--source-checkout", type=Path, required=True)
    build.add_argument("--out", type=Path, required=True)
    build.add_argument("--agent-image", required=True)
    build.add_argument("--agent-image-id", required=True)
    validate = sub.add_parser("validate")
    validate.add_argument("--root", type=Path, required=True)
    validate.add_argument("--source-release", type=Path, required=True)
    args = vars(parser.parse_args(argv))
    command = args.pop("command")
    try:
        result = build_release(out_dir=args.pop("out"), **args) if command == "build" else validate_release(**args)
    except (OSError, ValueError, Phase31BenchmarkError) as exc:
        print(f"PHASE31_BENCHMARK_ERROR {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
