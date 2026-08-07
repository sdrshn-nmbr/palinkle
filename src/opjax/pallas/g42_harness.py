"""DeepSWE-style task, workspace, trajectory, and grading contracts for G4.2."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import tomli

from opjax.pallas.environment import verify_static
from opjax.pallas.task_semantics import operation_specification, render_task_instruction

SCHEMA_VERSION = 1
TASK_SCHEMA_VERSION = "1.4"
LEGACY_TASK_SCHEMA_VERSION = "1.3"
AGENT_IMAGE = "python@sha256:9e869b0816f5537709825b49e62dc86d1c2691eff19b05c1d4dc3a07992cc052"
ACTION_PATTERN = re.compile(r"```mswea_bash_command\s*\n(.*?)\n```", re.DOTALL)
MANDATORY_STAGES = (
    "artifact_contract",
    "pallas_api",
    "tpu_compile",
    "full_shape_correctness",
    "normal_lowering",
    "runtime_safety",
    "profile",
)
FORBIDDEN_WORKSPACE_NAMES = {"tests", "solution"}


class G42HarnessError(RuntimeError):
    """The G4.2 harness cannot preserve its isolation or evidence contract."""


class AgentDriver(Protocol):
    """Provider-neutral boundary used by the G4.2 task runner."""

    def run(
        self,
        *,
        task_dir: Path,
        output_dir: Path,
        checkpoint: str | None,
        seed: int,
        turn_limit: int,
        snapshot_turns: tuple[int, ...],
    ) -> dict[str, Any]: ...


@dataclass(frozen=True)
class TaskPackage:
    root: Path
    task_id: str
    split: str
    mode: str
    family: str
    source_row_id: str
    mutation: str
    task_sha256: str
    input_shapes: tuple[tuple[int, ...], ...]
    input_dtypes: tuple[str, ...]
    correctness_seeds: tuple[int, ...]
    exact_semantics: bool


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_sha256(root: Path, *, excluded: set[str] | None = None) -> str:
    excluded = excluded or set()
    entries: list[dict[str, str]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or any(part in excluded for part in path.relative_to(root).parts):
            continue
        entries.append({"path": str(path.relative_to(root)), "sha256": file_sha256(path)})
    return canonical_sha256(entries)


def _require(condition: bool, code: str, detail: str) -> None:
    if not condition:
        raise G42HarnessError(f"{code}: {detail}")


def parse_action(content: str) -> dict[str, str]:
    actions = [match.strip() for match in ACTION_PATTERN.findall(content)]
    if len(actions) != 1:
        raise G42HarnessError(f"ACTION_COUNT_INVALID: expected=1 observed={len(actions)}")
    _require(bool(actions[0]), "ACTION_EMPTY", "command")
    return {"command": actions[0]}


def _tool_call_value(tool_call: Any) -> dict[str, Any]:
    if hasattr(tool_call, "model_dump"):
        value = tool_call.model_dump(mode="json")
    elif isinstance(tool_call, dict):
        value = tool_call
    else:
        raise G42HarnessError("ACTION_TOOL_INVALID: unsupported tool-call value")
    return value


def parse_model_action(message: dict[str, Any]) -> dict[str, str]:
    """Normalize one native TML tool call or one legacy fenced shell action."""
    text = model_message_text(message)
    fenced = [match.strip() for match in ACTION_PATTERN.findall(text)]
    native: list[dict[str, str]] = []
    for raw in message.get("tool_calls", ()) or ():
        tool_call = _tool_call_value(raw)
        function = tool_call.get("function")
        if not isinstance(function, dict):
            raise G42HarnessError("ACTION_TOOL_INVALID: function")
        if function.get("name") != "mswea_bash_command":
            raise G42HarnessError(
                f"ACTION_TOOL_INVALID: name={function.get('name')!r}"
            )
        arguments = function.get("arguments")
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError as exc:
                raise G42HarnessError("ACTION_ARGUMENTS_INVALID: malformed JSON") from exc
        if not isinstance(arguments, dict) or not isinstance(arguments.get("command"), str):
            raise G42HarnessError("ACTION_ARGUMENTS_INVALID: command")
        native.append({"command": arguments["command"].strip()})
    observed = len(fenced) + len(native)
    if observed != 1:
        raise G42HarnessError(f"ACTION_COUNT_INVALID: expected=1 observed={observed}")
    action = native[0] if native else {"command": fenced[0]}
    _require(bool(action["command"]), "ACTION_EMPTY", "command")
    return action


def model_message_text(message: dict[str, Any]) -> str:
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    return "\n".join(
        part.get("text", "")
        for part in content
        if isinstance(part, dict) and part.get("type") in {"text", "thinking"}
    )


def validate_horizon_contract(
    *, turn_limit: int, snapshot_turns: tuple[int, ...]
) -> None:
    valid = (
        turn_limit > 0
        and bool(snapshot_turns)
        and tuple(sorted(set(snapshot_turns))) == snapshot_turns
        and snapshot_turns[-1] <= turn_limit
    )
    _require(
        valid,
        "HORIZON_CONTRACT_INVALID",
        f"limit={turn_limit} snapshots={snapshot_turns}",
    )


def load_task_package(root: Path) -> TaskPackage:
    root = root.resolve()
    manifest_path = root / "task.toml"
    try:
        manifest = tomli.loads(manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise G42HarnessError(f"TASK_MANIFEST_MISSING: {manifest_path}") from exc
    except tomli.TOMLDecodeError as exc:
        raise G42HarnessError(f"TASK_MANIFEST_INVALID: {exc}") from exc
    schema_version = manifest.get("schema_version")
    _require(
        schema_version in {LEGACY_TASK_SCHEMA_VERSION, TASK_SCHEMA_VERSION},
        "TASK_SCHEMA_INVALID",
        root.name,
    )
    task = manifest.get("task", {})
    metadata = manifest.get("metadata", {})
    verifier = manifest.get("verifier", {})
    agent = manifest.get("agent", {})
    required_files = (
        "instruction.md",
        "pre_artifacts.sh",
        "environment/Dockerfile",
        "environment/starter/kernel.py",
        "environment/public/dev_check.py",
        "environment/public/PALLAS_API.md",
        "tests/task.json",
        "tests/test.sh",
        "solution/kernel.py",
    )
    for relative in required_files:
        _require((root / relative).is_file(), "TASK_FILE_MISSING", f"{root.name}:{relative}")
    _require(task.get("name") == f"opjax/{root.name}", "TASK_NAME_INVALID", repr(task.get("name")))
    _require(metadata.get("task_id") == root.name, "TASK_ID_INVALID", repr(metadata.get("task_id")))
    _require(metadata.get("split") in {"train", "near_heldout"}, "TASK_SPLIT_INVALID", root.name)
    _require(metadata.get("mode") in {"curriculum", "benchmark"}, "TASK_MODE_INVALID", root.name)
    if metadata["mode"] == "benchmark":
        _require(metadata["split"] == "near_heldout", "BENCHMARK_SPLIT_INVALID", root.name)
    else:
        _require(metadata["split"] == "train", "CURRICULUM_SPLIT_INVALID", root.name)
    _require(agent.get("network_mode") == "no-network", "AGENT_NETWORK_INVALID", root.name)
    _require(verifier.get("network_mode") == "no-network", "VERIFIER_NETWORK_INVALID", root.name)
    _require(verifier.get("environment_mode") == "separate", "VERIFIER_ISOLATION_INVALID", root.name)
    task_json = json.loads((root / "tests" / "task.json").read_text(encoding="utf-8"))
    _require(task_json.get("task_id") == root.name, "VERIFIER_TASK_ID_INVALID", root.name)
    _require(
        task_json.get("reference_kernel_sha256") == file_sha256(root / "solution" / "kernel.py"),
        "REFERENCE_KERNEL_HASH_MISMATCH",
        root.name,
    )
    seeds = tuple(task_json.get("correctness_seeds", ()))
    _require(seeds == (0, 1, 2), "CORRECTNESS_SEEDS_INVALID", repr(seeds))
    hashes = {
        relative: file_sha256(root / relative)
        for relative in required_files
        if relative != "task.toml"
    }
    hash_manifest = json.loads(json.dumps(manifest))
    hash_manifest["metadata"]["task_sha256"] = None
    observed_sha = canonical_sha256({"manifest": hash_manifest, "files": hashes})
    declared_sha = metadata.get("task_sha256")
    _require(declared_sha == observed_sha, "TASK_HASH_MISMATCH", f"{declared_sha} != {observed_sha}")
    exact_semantics = schema_version == TASK_SCHEMA_VERSION
    if exact_semantics:
        specification = operation_specification(task_json)
        _require(
            task_json.get("public_specification") == specification,
            "PUBLIC_SPECIFICATION_MISMATCH",
            root.name,
        )
        _require(
            task_json.get("public_specification_sha256")
            == canonical_sha256(specification),
            "PUBLIC_SPECIFICATION_HASH_MISMATCH",
            root.name,
        )
        expected_instruction = render_task_instruction(
            task_json,
            repair=metadata["mutation"] if metadata["mode"] == "curriculum" else None,
        )
        _require(
            (root / "instruction.md").read_text(encoding="utf-8")
            == expected_instruction,
            "PUBLIC_INSTRUCTION_MISMATCH",
            root.name,
        )
    return TaskPackage(
        root=root,
        task_id=root.name,
        split=metadata["split"],
        mode=metadata["mode"],
        family=metadata["family"],
        source_row_id=metadata["source_row_id"],
        mutation=metadata["mutation"],
        task_sha256=observed_sha,
        input_shapes=tuple(tuple(shape) for shape in task_json["input_shapes"]),
        input_dtypes=tuple(task_json["input_dtypes"]),
        correctness_seeds=seeds,
        exact_semantics=exact_semantics,
    )


def validate_task_release(root: Path) -> dict[str, Any]:
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _require(manifest.get("kind") == "pallas_g42_task_release", "RELEASE_KIND_INVALID", str(root))
    packages = [load_task_package(root / relative) for relative in manifest.get("tasks", [])]
    _require(len(packages) == manifest.get("counts", {}).get("pool"), "RELEASE_POOL_COUNT_INVALID", str(root))
    selected = manifest.get("training_selection", [])
    _require(len(selected) == 32 and len(set(selected)) == 32, "TRAIN_SELECTION_INVALID", repr(len(selected)))
    by_id = {package.task_id: package for package in packages}
    _require(set(selected) <= set(by_id), "TRAIN_SELECTION_UNKNOWN", str(root))
    families: dict[str, int] = {}
    for task_id in selected:
        package = by_id[task_id]
        _require(package.mode == "curriculum", "TRAIN_SELECTION_MODE_INVALID", task_id)
        families[package.family] = families.get(package.family, 0) + 1
    _require(len(families) == 8, "TRAIN_FAMILY_COUNT_INVALID", repr(families))
    _require(set(families.values()) == {4}, "TRAIN_FAMILY_DEPTH_INVALID", repr(families))
    observed = canonical_sha256(
        {
            "tasks": [{"task_id": package.task_id, "task_sha256": package.task_sha256} for package in packages],
            "training_selection": selected,
            "source_release_sha256": manifest.get("source_release_sha256"),
        }
    )
    _require(manifest.get("release_sha256") == observed, "RELEASE_HASH_MISMATCH", str(root))
    return {"task_count": len(packages), "training_count": len(selected), "families": families, "release_sha256": observed}


def create_agent_workspace(task: TaskPackage, destination: Path) -> dict[str, Any]:
    _require(not destination.exists(), "WORKSPACE_EXISTS", str(destination))
    destination.mkdir(parents=True)
    shutil.copy2(task.root / "instruction.md", destination / "instruction.md")
    shutil.copy2(task.root / "environment" / "starter" / "kernel.py", destination / "kernel.py")
    shutil.copy2(task.root / "environment" / "public" / "dev_check.py", destination / "dev_check.py")
    shutil.copy2(task.root / "environment" / "public" / "PALLAS_API.md", destination / "PALLAS_API.md")
    for forbidden in FORBIDDEN_WORKSPACE_NAMES:
        _require(not (destination / forbidden).exists(), "HIDDEN_PATH_EXPOSED", forbidden)
    environment = os.environ.copy()
    environment.update(
        {
            "GIT_AUTHOR_NAME": "opjax-harness",
            "GIT_AUTHOR_EMAIL": "harness@opjax.invalid",
            "GIT_COMMITTER_NAME": "opjax-harness",
            "GIT_COMMITTER_EMAIL": "harness@opjax.invalid",
        }
    )
    subprocess.run(["git", "init", "-q", str(destination)], check=True, env=environment)
    subprocess.run(["git", "-C", str(destination), "add", "."], check=True, env=environment)
    subprocess.run(["git", "-C", str(destination), "commit", "-q", "-m", "task base"], check=True, env=environment)
    base_commit = subprocess.run(
        ["git", "-C", str(destination), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return {"base_commit": base_commit, "workspace_sha256": tree_sha256(destination, excluded={".git"})}


def snapshot_workspace(workspace: Path, *, turn: int, output_dir: Path) -> dict[str, Any]:
    _require(turn > 0, "TURN_INVALID", str(turn))
    output_dir.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment.update(
        {
            "GIT_AUTHOR_NAME": "opjax-harness",
            "GIT_AUTHOR_EMAIL": "harness@opjax.invalid",
            "GIT_COMMITTER_NAME": "opjax-harness",
            "GIT_COMMITTER_EMAIL": "harness@opjax.invalid",
        }
    )
    subprocess.run(["git", "-C", str(workspace), "add", "-A"], check=True, env=environment)
    subprocess.run(
        ["git", "-C", str(workspace), "commit", "-q", "--allow-empty", "-m", f"turn {turn}"],
        check=True,
        env=environment,
    )
    base = subprocess.run(
        ["git", "-C", str(workspace), "rev-list", "--max-parents=0", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    patch = subprocess.run(
        ["git", "-C", str(workspace), "diff", "--binary", base, "HEAD"],
        check=True,
        capture_output=True,
    ).stdout
    patch_path = output_dir / f"turn-{turn}.patch"
    patch_path.write_bytes(patch)
    kernel_path = output_dir / f"turn-{turn}-kernel.py"
    workspace_kernel = workspace / "kernel.py"
    if workspace_kernel.is_file() and not workspace_kernel.is_symlink():
        kernel_path.write_bytes(workspace_kernel.read_bytes())
    else:
        kernel_path.write_bytes(b"")
    record = {
        "turn": turn,
        "commit": subprocess.run(
            ["git", "-C", str(workspace), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "patch_path": patch_path.name,
        "patch_sha256": file_sha256(patch_path),
        "kernel_path": kernel_path.name,
        "kernel_sha256": file_sha256(kernel_path),
        "workspace_sha256": tree_sha256(workspace, excluded={".git"}),
    }
    (output_dir / f"turn-{turn}.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return record


def materialize_submission(*, task: TaskPackage, patch_path: Path, destination: Path) -> dict[str, Any]:
    """Apply only a captured agent patch to a fresh copy of the immutable task base."""
    workspace_record = create_agent_workspace(task, destination)
    patch_bytes = patch_path.read_bytes()
    patch_sha256 = hashlib.sha256(patch_bytes).hexdigest()
    if patch_bytes:
        applied = subprocess.run(
            ["git", "-C", str(destination), "apply", "--whitespace=error-all", "-"],
            input=patch_bytes,
            capture_output=True,
        )
        if applied.returncode != 0:
            raise G42HarnessError(
                f"PATCH_APPLY_FAILED: {applied.stderr.decode(errors='replace').strip()}"
            )
    kernel_path = destination / "kernel.py"
    _require(kernel_path.is_file() and not kernel_path.is_symlink(), "KERNEL_ARTIFACT_INVALID", str(kernel_path))
    for forbidden in FORBIDDEN_WORKSPACE_NAMES:
        _require(not (destination / forbidden).exists(), "PATCH_CREATED_HIDDEN_PATH", forbidden)
    return {
        **workspace_record,
        "patch_sha256": patch_sha256,
        "kernel_sha256": file_sha256(kernel_path),
        "workspace_sha256": tree_sha256(destination, excluded={".git"}),
        "kernel_path": str(kernel_path),
    }


def static_dev_result(kernel_path: Path) -> dict[str, Any]:
    source = kernel_path.read_text(encoding="utf-8")
    verdict = verify_static(f"```python\n{source}\n```")
    return {
        "passed": verdict.passed,
        "stage": verdict.stage,
        "feedback": verdict.feedback,
        "evidence": verdict.evidence,
    }


def classify_verifier_result(result: dict[str, Any]) -> int:
    if result.get("infrastructure_error") is True:
        return -1
    stages = result.get("stages")
    if not isinstance(stages, dict):
        return 0
    mandatory_passed = all(stages.get(stage) is True for stage in MANDATORY_STAGES)
    profile = result.get("profile")
    profile_admitted = bool(
        isinstance(profile, dict)
        and isinstance(profile.get("admission"), dict)
        and profile["admission"].get("verified") is True
    )
    return int(
        result.get("passed") is True
        and result.get("stage") == "verified"
        and mandatory_passed
        and profile_admitted
    )


def write_verifier_artifacts(
    *,
    result: dict[str, Any],
    output_dir: Path,
    task_id: str,
    kernel_sha256: str,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    reward = classify_verifier_result(result)
    reached = result.get("stages", {})
    stage_scores = {stage: float(reached.get(stage, False)) for stage in MANDATORY_STAGES}
    profile = result.get("profile") or {}
    timing = profile.get("timing") if isinstance(profile.get("timing"), dict) else {}
    speedup = timing.get("speedup", profile.get("speedup"))
    profile_admitted = bool(
        reached.get("profile") is True
        and isinstance(profile.get("admission"), dict)
        and profile["admission"].get("verified") is True
    )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "task_id": task_id,
        "kernel_sha256": kernel_sha256,
        "reward": reward,
        "stage_fractions": stage_scores,
        "correct": bool(result.get("correct", result.get("passed", False))),
        "authentic": bool(result.get("authentic", result.get("passed", False))),
        "normal_lowered": bool(result.get("normal_lowered", result.get("passed", False))),
        "profiled": profile_admitted,
        "speedup": speedup,
        "beats_xla": bool(
            profile_admitted and timing.get("materially_beats_xla") is True
        ),
        "failure_stage": None if reward == 1 else result.get("stage", "infrastructure"),
        "infrastructure_error": reward == -1,
        "worker_recovery_required": bool(result.get("worker_recovery_required", False)),
    }
    cases = []
    failed_stage = payload["failure_stage"]
    for stage in MANDATORY_STAGES:
        passed = stage_scores[stage] == 1.0
        status = "passed" if passed else "failed" if stage == failed_stage else "not_run"
        cases.append({"name": stage, "status": status, "message": result.get("error") if status == "failed" else None})
    (output_dir / "reward.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "ctrf.json").write_text(
        json.dumps({"schema_version": SCHEMA_VERSION, "task_id": task_id, "tests": cases}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    log = json.dumps(result, indent=2, sort_keys=True) + "\n"
    (output_dir / "run.log").write_text(log, encoding="utf-8")
    (output_dir / "test-stdout.txt").write_text(log, encoding="utf-8")
    return payload


def summarize_horizons(rows: list[dict[str, Any]]) -> dict[str, Any]:
    keyed: dict[tuple[str, str, int], dict[int, dict[str, Any]]] = {}
    for row in rows:
        key = (row["model_id"], row["task_id"], int(row["seed"]))
        keyed.setdefault(key, {})[int(row["turn"])] = row
    transitions = {"fail_to_pass": 0, "pass_to_pass": 0, "fail_to_fail": 0, "pass_to_fail": 0}
    transitions_by_model: dict[str, dict[str, int]] = {}
    by_model: dict[str, dict[str, int]] = {}
    for (model_id, _, _), horizons in keyed.items():
        _require(set(horizons) == {3, 6}, "HORIZON_PAIR_INCOMPLETE", repr((model_id, horizons.keys())))
        before = horizons[3]["reward"] == 1
        after = horizons[6]["reward"] == 1
        transition = f"{'pass' if before else 'fail'}_to_{'pass' if after else 'fail'}"
        transitions[transition] += 1
        model_transitions = transitions_by_model.setdefault(
            model_id,
            {"fail_to_pass": 0, "pass_to_pass": 0, "fail_to_fail": 0, "pass_to_fail": 0},
        )
        model_transitions[transition] += 1
        counts = by_model.setdefault(model_id, {"k3_verified": 0, "k6_verified": 0})
        counts["k3_verified"] += int(before)
        counts["k6_verified"] += int(after)
    return {
        "paired_units": len(keyed),
        "transitions": transitions,
        "transitions_by_model": transitions_by_model,
        "models": by_model,
    }
