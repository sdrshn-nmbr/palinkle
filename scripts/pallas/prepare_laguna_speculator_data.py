from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
from typing import Any

from opjax.pallas.laguna_dspark_conformance import canonical_sha256
from opjax.pallas.laguna_speculative import BASH_TOOL


_RUN = re.compile(r"^(?P<model>.+)--(?P<task>.+)--seed-(?P<seed>\d+)$")
MAX_TRAINING_CONTEXT = 18_432
_MESSAGE_KEYS = (
    "role",
    "content",
    "reasoning_content",
    "tool_calls",
    "tool_call_id",
    "name",
)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _public_message(message: dict[str, Any]) -> dict[str, Any]:
    result = {key: message[key] for key in _MESSAGE_KEYS if key in message}
    calls = result.get("tool_calls")
    if not isinstance(calls, list):
        return result
    normalized_calls = []
    for call in calls:
        if not isinstance(call, dict):
            raise ValueError("LAGUNA_TRAINING_TOOL_CALL_INVALID")
        normalized = dict(call)
        function = normalized.get("function")
        if not isinstance(function, dict):
            raise ValueError("LAGUNA_TRAINING_TOOL_FUNCTION_INVALID")
        normalized_function = dict(function)
        arguments = normalized_function.get("arguments")
        if isinstance(arguments, str):
            normalized_function["arguments"] = json.loads(arguments)
        if not isinstance(normalized_function.get("arguments"), dict):
            raise ValueError("LAGUNA_TRAINING_TOOL_ARGUMENTS_INVALID")
        normalized["function"] = normalized_function
        normalized_calls.append(normalized)
    result["tool_calls"] = normalized_calls
    return result


def build_rows(
    sample_root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    train: list[dict[str, Any]] = []
    calibration: list[dict[str, Any]] = []
    heldout: list[dict[str, Any]] = []
    trajectories = sorted(sample_root.glob("runs/*/trajectory.json"))
    if not trajectories:
        raise ValueError(f"LAGUNA_TRAINING_TRAJECTORIES_MISSING:{sample_root}")
    parsed = []
    for path in trajectories:
        match = _RUN.fullmatch(path.parent.name)
        if match is None:
            raise ValueError(f"LAGUNA_TRAJECTORY_ID_INVALID:{path}")
        seed = int(match.group("seed"))
        if seed not in {0, 1, 2}:
            raise ValueError(f"LAGUNA_TRAJECTORY_SEED_INVALID:{path}:{seed}")
        parsed.append((path, match.group("task"), seed))
    tasks = sorted({task for _, task, _ in parsed})
    reserved_tasks = tasks[::4]
    calibration_tasks = set(reserved_tasks[::2])
    heldout_tasks = set(reserved_tasks[1::2])
    for path, task, seed in parsed:
        payload = json.loads(path.read_text(encoding="utf-8"))
        messages = payload.get("messages")
        if not isinstance(messages, list) or not messages:
            raise ValueError(f"LAGUNA_TRAJECTORY_MESSAGES_INVALID:{path}")
        public: list[dict[str, Any]] = []
        for message in messages:
            if not isinstance(message, dict):
                raise ValueError(f"LAGUNA_TRAJECTORY_MESSAGE_INVALID:{path}")
            public.append(_public_message(message))
        assistant_calls = sum(message.get("role") == "assistant" for message in messages)
        if assistant_calls == 0:
            raise ValueError(f"LAGUNA_TRAJECTORY_ASSISTANT_MISSING:{path}")
        row = {
            "id": path.parent.name,
            "trajectory": path.parent.name,
            "task": task,
            "seed": seed,
            "assistant_calls": assistant_calls,
            "conversations": public,
            "tools": [BASH_TOOL],
        }
        destination = (
            calibration
            if task in calibration_tasks
            else heldout
            if task in heldout_tasks
            else train
        )
        destination.append(row)
    if not train or not calibration or not heldout:
        raise ValueError("LAGUNA_TRAINING_SPLIT_EMPTY")
    split_trajectories = [
        {row["trajectory"] for row in rows}
        for rows in (train, calibration, heldout)
    ]
    if any(
        left & right
        for index, left in enumerate(split_trajectories)
        for right in split_trajectories[index + 1 :]
    ):
        raise ValueError("LAGUNA_TRAINING_TRAJECTORY_LEAKAGE")
    return train, calibration, heldout


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sample-root",
        type=Path,
        default=Path("data/pallas/runs/phase32-base-capability/laguna-samples"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/pallas/corpora/laguna-speculator-v1"),
    )
    args = parser.parse_args()
    train, calibration, heldout = build_rows(args.sample_root.resolve())
    args.output_root.mkdir(parents=True, exist_ok=False)
    paths = {
        "train": args.output_root / "train.jsonl",
        "calibration": args.output_root / "calibration.jsonl",
        "heldout": args.output_root / "heldout.jsonl",
    }
    _write_jsonl(paths["train"], train)
    _write_jsonl(paths["calibration"], calibration)
    _write_jsonl(paths["heldout"], heldout)
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "kind": "opjax_laguna_speculator_training_corpus",
        "source_root": str(args.sample_root.resolve()),
        "split_policy": (
            "one_complete_trajectory_per_row; sorted_task_id_every_fourth_reserved; "
            "alternating_reserved_tasks_calibration_and_final_heldout"
        ),
        "max_training_context": MAX_TRAINING_CONTEXT,
        "tool_schema_sha256": canonical_sha256(BASH_TOOL),
        "rows": {
            "train": len(train),
            "calibration": len(calibration),
            "heldout": len(heldout),
        },
        "trajectories": {
            "train": len({row["trajectory"] for row in train}),
            "calibration": len({row["trajectory"] for row in calibration}),
            "heldout": len({row["trajectory"] for row in heldout}),
        },
        "tasks": {
            "train": len({row["task"] for row in train}),
            "calibration": len({row["task"] for row in calibration}),
            "heldout": len({row["task"] for row in heldout}),
        },
        "files": {
            split: {
                "path": path.name,
                "sha256": _file_sha256(path),
                "bytes": path.stat().st_size,
            }
            for split, path in paths.items()
        },
    }
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    (args.output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
