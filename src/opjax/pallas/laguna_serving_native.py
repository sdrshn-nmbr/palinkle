from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from opjax.pallas.laguna_dspark_conformance import canonical_sha256


TARGET_FEATURE_WIDTH = 10_240
TARGET_HIDDEN_WIDTH = 2_048
MIN_LOSS_TOKENS = 14


class ServingNativeError(ValueError):
    pass


def serving_native_sample_dirs(capture_root: Path, split: str) -> list[Path]:
    paths = sorted(
        capture_root.glob(
            "rebuilt-fixed-shards/"
            f"{split}-*/policies/first-causal-observation-wins/samples/{split}/*"
        ),
        key=lambda path: path.name,
    )
    if not paths:
        raise ServingNativeError(
            f"SERVING_NATIVE_CACHE_SAMPLES_MISSING:{capture_root}:{split}"
        )
    prompt_ids = [path.name for path in paths]
    if len(prompt_ids) != len(set(prompt_ids)):
        raise ServingNativeError(f"SERVING_NATIVE_CACHE_SAMPLE_DUPLICATE:{split}")
    return paths


def _as_bfloat16_bits(value: np.ndarray) -> np.ndarray:
    if value.dtype == np.uint16:
        return value
    if value.dtype != np.float32:
        raise ServingNativeError(f"SERVING_NATIVE_HIDDEN_DTYPE_INVALID:{value.dtype}")
    bits = value.view(np.uint32)
    rounding_bias = np.uint32(0x7FFF) + ((bits >> 16) & np.uint32(1))
    return ((bits + rounding_bias) >> 16).astype(np.uint16)


def _causal_prefix_length(
    token_ids: np.ndarray,
    positions: np.ndarray,
    committed: np.ndarray,
) -> int:
    if len(token_ids) != len(positions) or not len(positions):
        raise ServingNativeError("SERVING_NATIVE_CAUSAL_PREFIX_SHAPE_INVALID")
    if len(positions) > 1 and not np.array_equal(
        np.diff(positions.astype(np.int64)), np.ones(len(positions) - 1, dtype=np.int64)
    ):
        raise ServingNativeError("SERVING_NATIVE_CAUSAL_POSITIONS_INVALID")
    for index, position in enumerate(positions.astype(np.int64).tolist()):
        if (
            position < 0
            or position >= len(committed)
            or int(token_ids[index]) != int(committed[position])
        ):
            return index
    return len(positions)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def select_final_requests(
    replay_corpus: dict[str, Any], split_manifest: dict[str, Any]
) -> dict[str, list[dict[str, Any]]]:
    records = replay_corpus.get("records")
    task_ids = split_manifest.get("task_ids")
    if not isinstance(records, list) or not isinstance(task_ids, dict):
        raise ServingNativeError("SERVING_NATIVE_CORPUS_INVALID")
    task_to_split = {
        task: split
        for split in ("train", "calibration", "heldout")
        for task in task_ids.get(split, [])
    }
    if len(task_to_split) != sum(len(task_ids.get(split, [])) for split in task_ids):
        raise ServingNativeError("SERVING_NATIVE_TASK_SPLIT_OVERLAP")
    final_by_trajectory: dict[str, dict[str, Any]] = {}
    for record in records:
        trajectory = record.get("trajectory")
        call = record.get("call")
        if not isinstance(trajectory, str) or not isinstance(call, int):
            raise ServingNativeError("SERVING_NATIVE_REPLAY_RECORD_INVALID")
        previous = final_by_trajectory.get(trajectory)
        if previous is None or call > previous["call"]:
            final_by_trajectory[trajectory] = record
    result = {split: [] for split in ("train", "calibration", "heldout")}
    for trajectory, record in sorted(final_by_trajectory.items()):
        matches = [
            task
            for task in task_to_split
            if f"--{task}--seed-" in trajectory
        ]
        if len(matches) != 1:
            raise ServingNativeError(
                f"SERVING_NATIVE_TRAJECTORY_TASK_INVALID:{trajectory}:{matches}"
            )
        split = task_to_split[matches[0]]
        result[split].append(record)
    expected = split_manifest.get("trajectories")
    if not isinstance(expected, dict) or any(
        len(result[split]) != int(expected[split]) for split in result
    ):
        raise ServingNativeError(
            "SERVING_NATIVE_SPLIT_COUNT_INVALID:"
            f"{ {key: len(value) for key, value in result.items()} }:{expected}"
        )
    trajectories = [
        {row["trajectory"] for row in result[split]} for split in result
    ]
    if any(
        left & right
        for index, left in enumerate(trajectories)
        for right in trajectories[index + 1 :]
    ):
        raise ServingNativeError("SERVING_NATIVE_TRAJECTORY_LEAKAGE")
    return result


def _load_ledger(session_root: Path) -> dict[int, dict[str, np.ndarray]]:
    ledger = session_root / "target-ledger.jsonl"
    if not ledger.is_file():
        raise ServingNativeError(f"SERVING_NATIVE_LEDGER_MISSING:{ledger}")
    rounds: dict[int, dict[str, np.ndarray]] = {}
    for line in ledger.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        round_id = record.get("round")
        name = record.get("name")
        path = session_root / str(record.get("path"))
        if not isinstance(round_id, int) or not isinstance(name, str):
            raise ServingNativeError("SERVING_NATIVE_LEDGER_RECORD_INVALID")
        if not path.is_file() or _file_sha256(path) != record.get("sha256"):
            raise ServingNativeError(f"SERVING_NATIVE_TENSOR_HASH_INVALID:{path}")
        value = np.load(path, allow_pickle=False)
        expected = {
            "target_input_ids": ("int32", "torch.int32"),
            "target_positions": ("int64", "torch.int64"),
            "target_aux_hidden_states": ("uint16", "torch.bfloat16"),
            "target_last_hidden_states": ("uint16", "torch.bfloat16"),
        }
        if name not in expected:
            raise ServingNativeError(f"SERVING_NATIVE_BOUNDARY_NAME_INVALID:{name}")
        if (
            str(value.dtype) != record.get("dtype")
            or record.get("dtype") != expected[name][0]
            or record.get("source_dtype") != expected[name][1]
            or list(value.shape) != record.get("shape")
        ):
            raise ServingNativeError(f"SERVING_NATIVE_TENSOR_CONTRACT_INVALID:{path}")
        cell = rounds.setdefault(round_id, {})
        if name in cell:
            raise ServingNativeError(
                f"SERVING_NATIVE_BOUNDARY_DUPLICATE:{round_id}:{name}"
            )
        cell[name] = value
    if sorted(rounds) != list(range(len(rounds))):
        raise ServingNativeError(f"SERVING_NATIVE_ROUNDS_INVALID:{sorted(rounds)}")
    return rounds


def captured_prompt_token_ids(
    *, session_root: Path, prompt_token_count: int
) -> list[int]:
    if prompt_token_count < 1:
        raise ServingNativeError(
            f"SERVING_NATIVE_PROMPT_TOKEN_COUNT_INVALID:{prompt_token_count}"
        )
    first = _load_ledger(session_root).get(0, {})
    if "target_input_ids" not in first or "target_positions" not in first:
        raise ServingNativeError("SERVING_NATIVE_PROMPT_BOUNDARIES_MISSING")
    token_ids = first["target_input_ids"].reshape(-1).astype(np.int64)
    positions = first["target_positions"].reshape(-1).astype(np.int64)
    result = np.full(prompt_token_count, -1, dtype=np.int64)
    seen = np.zeros(prompt_token_count, dtype=np.bool_)
    for token_id, position in zip(token_ids.tolist(), positions.tolist()):
        if 0 <= position < prompt_token_count:
            if seen[position]:
                raise ServingNativeError(
                    f"SERVING_NATIVE_PROMPT_POSITION_DUPLICATE:{position}"
                )
            result[position] = token_id
            seen[position] = True
    if not bool(seen.all()):
        missing = np.flatnonzero(~seen)
        raise ServingNativeError(
            "SERVING_NATIVE_PROMPT_COVERAGE_MISSING:"
            + ",".join(str(value) for value in missing[:32])
        )
    return result.tolist()


def reconstruct_committed_sample(
    *,
    session_root: Path,
    prompt_token_ids: list[int],
    completion_token_ids: list[int],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    committed = np.asarray([*prompt_token_ids, *completion_token_ids], dtype=np.int64)
    if committed.ndim != 1 or not len(prompt_token_ids) or not len(completion_token_ids):
        raise ServingNativeError("SERVING_NATIVE_COMMITTED_TOKENS_INVALID")
    features = np.empty((len(committed), TARGET_FEATURE_WIDTH), dtype=np.uint16)
    last_hidden = np.empty((len(committed), TARGET_HIDDEN_WIDTH), dtype=np.uint16)
    covered = np.zeros(len(committed), dtype=np.bool_)
    source_round = np.full(len(committed), -1, dtype=np.int32)
    rejected_rows = 0
    rounds = _load_ledger(session_root)
    required = {
        "target_input_ids",
        "target_positions",
        "target_aux_hidden_states",
        "target_last_hidden_states",
    }
    for round_id, boundaries in rounds.items():
        if set(boundaries) != required:
            raise ServingNativeError(
                f"SERVING_NATIVE_BOUNDARIES_INVALID:{round_id}:{sorted(boundaries)}"
            )
        token_ids = boundaries["target_input_ids"].reshape(-1).astype(np.int64)
        positions = boundaries["target_positions"].reshape(-1).astype(np.int64)
        aux = _as_bfloat16_bits(
            boundaries["target_aux_hidden_states"].reshape(-1, TARGET_FEATURE_WIDTH)
        )
        final = _as_bfloat16_bits(
            boundaries["target_last_hidden_states"].reshape(-1, TARGET_HIDDEN_WIDTH)
        )
        if not (len(token_ids) == len(positions) == len(aux) == len(final)):
            raise ServingNativeError(
                f"SERVING_NATIVE_ROUND_LENGTH_INVALID:{round_id}:"
                f"{len(token_ids)}:{len(positions)}:{len(aux)}:{len(final)}"
            )
        causal_length = _causal_prefix_length(token_ids, positions, committed)
        rejected_rows += len(positions) - causal_length
        for index, position in enumerate(positions[:causal_length].tolist()):
            features[position] = aux[index]
            last_hidden[position] = final[index]
            covered[position] = True
            source_round[position] = round_id
    missing = np.flatnonzero(~covered)
    if missing.size:
        raise ServingNativeError(
            "SERVING_NATIVE_COMMITTED_COVERAGE_MISSING:"
            + ",".join(str(value) for value in missing[:32])
        )
    loss_mask = np.zeros(len(committed), dtype=np.uint8)
    loss_mask[len(prompt_token_ids) :] = 1
    if int(loss_mask.sum()) < MIN_LOSS_TOKENS:
        raise ServingNativeError(
            f"SERVING_NATIVE_LOSS_TOKENS_INSUFFICIENT:{int(loss_mask.sum())}"
        )
    sample = {
        "input_ids": committed.astype(np.int32),
        "attention_mask": np.ones(len(committed), dtype=np.uint8),
        "loss_mask": loss_mask,
        "target_hidden_states": features,
        "target_last_hidden_states": last_hidden,
    }
    metadata: dict[str, Any] = {
        "schema_version": 1,
        "tokens": len(committed),
        "prompt_tokens": len(prompt_token_ids),
        "completion_tokens": len(completion_token_ids),
        "target_rounds": len(rounds),
        "rejected_or_out_of_range_rows": rejected_rows,
        "source_round_min": int(source_round.min()),
        "source_round_max": int(source_round.max()),
        "committed_token_sha256": hashlib.sha256(committed.tobytes()).hexdigest(),
    }
    metadata["metadata_sha256"] = canonical_sha256(metadata)
    return sample, metadata


def serving_prefix_ends(loss_mask: np.ndarray, *, proposal_length: int = 15) -> list[int]:
    mask = np.asarray(loss_mask).reshape(-1)
    if mask.size < 1 or not np.isin(mask, [0, 1]).all() or proposal_length < 1:
        raise ServingNativeError("SERVING_NATIVE_PREFIX_MASK_INVALID")
    ends = []
    start = 0
    while start < len(mask):
        value = int(mask[start])
        end = start + 1
        while end < len(mask) and int(mask[end]) == value:
            end += 1
        if value == 0:
            ends.append(end)
        else:
            ends.extend(range(start + proposal_length, end, proposal_length))
            ends.append(end)
        start = end
    result = sorted(set(ends))
    if result[-1] != len(mask):
        raise ServingNativeError("SERVING_NATIVE_PREFIX_END_INVALID")
    return result


def reconstruct_fixed_sample(
    *,
    session_root: Path,
    input_ids: np.ndarray,
    loss_mask: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    committed = np.asarray(input_ids, dtype=np.int64).reshape(-1)
    mask = np.asarray(loss_mask, dtype=np.uint8).reshape(-1)
    if committed.size < 1 or committed.shape != mask.shape or not np.isin(mask, [0, 1]).all():
        raise ServingNativeError("SERVING_NATIVE_FIXED_SAMPLE_INVALID")
    features = np.empty((len(committed), TARGET_FEATURE_WIDTH), dtype=np.uint16)
    last_hidden = np.empty((len(committed), TARGET_HIDDEN_WIDTH), dtype=np.uint16)
    covered = np.zeros(len(committed), dtype=np.bool_)
    source_round = np.full(len(committed), -1, dtype=np.int32)
    rejected_rows = 0
    overlap_rows = 0
    divergent_overlap_rows = 0
    divergent_loss_overlap_rows = 0
    round_receipts = []
    rounds = _load_ledger(session_root)
    required = {
        "target_input_ids",
        "target_positions",
        "target_aux_hidden_states",
        "target_last_hidden_states",
    }
    for round_id, boundaries in rounds.items():
        if set(boundaries) != required:
            raise ServingNativeError(
                f"SERVING_NATIVE_BOUNDARIES_INVALID:{round_id}:{sorted(boundaries)}"
            )
        token_ids = boundaries["target_input_ids"].reshape(-1).astype(np.int64)
        positions = boundaries["target_positions"].reshape(-1).astype(np.int64)
        aux = _as_bfloat16_bits(
            boundaries["target_aux_hidden_states"].reshape(-1, TARGET_FEATURE_WIDTH)
        )
        final = _as_bfloat16_bits(
            boundaries["target_last_hidden_states"].reshape(-1, TARGET_HIDDEN_WIDTH)
        )
        if not (len(token_ids) == len(positions) == len(aux) == len(final)):
            raise ServingNativeError(f"SERVING_NATIVE_ROUND_LENGTH_INVALID:{round_id}")
        causal_length = _causal_prefix_length(token_ids, positions, committed)
        rejected_rows += len(positions) - causal_length
        for index, position in enumerate(positions[:causal_length].tolist()):
            if covered[position]:
                overlap_rows += 1
                divergent = not (
                    np.array_equal(features[position], aux[index])
                    and np.array_equal(last_hidden[position], final[index])
                )
                divergent_overlap_rows += int(divergent)
                divergent_loss_overlap_rows += int(divergent and bool(mask[position]))
                continue
            features[position] = aux[index]
            last_hidden[position] = final[index]
            covered[position] = True
            source_round[position] = round_id
        round_receipts.append(
            {
                "round": round_id,
                "position_min": int(positions.min()),
                "position_max": int(positions.max()),
                "rows": len(positions),
                "committed_rows": causal_length,
            }
        )
    missing = np.flatnonzero(~covered)
    if missing.size:
        raise ServingNativeError(
            "SERVING_NATIVE_COMMITTED_COVERAGE_MISSING:"
            + ",".join(str(value) for value in missing[:32])
        )
    if int(mask.sum()) < MIN_LOSS_TOKENS:
        raise ServingNativeError(
            f"SERVING_NATIVE_LOSS_TOKENS_INSUFFICIENT:{int(mask.sum())}"
        )
    sample = {
        "input_ids": committed.astype(np.int32),
        "attention_mask": np.ones(len(committed), dtype=np.uint8),
        "loss_mask": mask,
        "target_hidden_states": features,
        "target_last_hidden_states": last_hidden,
    }
    metadata: dict[str, Any] = {
        "schema_version": 1,
        "mode": "fixed_arm_independent_prefix_replay",
        "tokens": len(committed),
        "loss_tokens": int(mask.sum()),
        "target_rounds": len(rounds),
        "rejected_or_out_of_range_rows": rejected_rows,
        "feature_selection_policy": "first_causal_observation_wins",
        "overlap_rows": overlap_rows,
        "divergent_overlap_rows": divergent_overlap_rows,
        "divergent_loss_overlap_rows": divergent_loss_overlap_rows,
        "source_round_min": int(source_round.min()),
        "source_round_max": int(source_round.max()),
        "committed_token_sha256": hashlib.sha256(committed.tobytes()).hexdigest(),
        "loss_mask_sha256": hashlib.sha256(mask.tobytes()).hexdigest(),
        "rounds": round_receipts,
    }
    metadata["metadata_sha256"] = canonical_sha256(metadata)
    return sample, metadata


def write_sample(
    output_root: Path,
    *,
    sample: dict[str, np.ndarray],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=False)
    files = {}
    for name, value in sample.items():
        path = output_root / f"{name}.npy"
        np.save(path, value, allow_pickle=False)
        files[name] = {
            "path": path.name,
            "sha256": _file_sha256(path),
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "kind": "opjax_laguna_serving_native_sample",
        "metadata": metadata,
        "files": files,
    }
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def validate_sample(output_root: Path) -> dict[str, Any]:
    manifest_path = output_root / "manifest.json"
    result_path = output_root / "capture-result.json"
    if not manifest_path.is_file() or not result_path.is_file():
        raise ServingNativeError(f"SERVING_NATIVE_SAMPLE_INCOMPLETE:{output_root}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if canonical_sha256(
        {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    ) != manifest.get("manifest_sha256"):
        raise ServingNativeError("SERVING_NATIVE_MANIFEST_HASH_INVALID")
    expected = {
        "input_ids": ([manifest["metadata"]["tokens"]], "int32"),
        "attention_mask": ([manifest["metadata"]["tokens"]], "uint8"),
        "loss_mask": ([manifest["metadata"]["tokens"]], "uint8"),
        "target_hidden_states": (
            [manifest["metadata"]["tokens"], TARGET_FEATURE_WIDTH],
            "uint16",
        ),
        "target_last_hidden_states": (
            [manifest["metadata"]["tokens"], TARGET_HIDDEN_WIDTH],
            "uint16",
        ),
    }
    if set(manifest.get("files", {})) != set(expected):
        raise ServingNativeError("SERVING_NATIVE_SAMPLE_FILES_INVALID")
    for name, (shape, dtype) in expected.items():
        record = manifest["files"][name]
        path = output_root / record["path"]
        if (
            record.get("shape") != shape
            or record.get("dtype") != dtype
            or not path.is_file()
            or _file_sha256(path) != record.get("sha256")
        ):
            raise ServingNativeError(f"SERVING_NATIVE_SAMPLE_FILE_INVALID:{name}")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if canonical_sha256(
        {key: value for key, value in result.items() if key != "result_sha256"}
    ) != result.get("result_sha256"):
        raise ServingNativeError("SERVING_NATIVE_RESULT_HASH_INVALID")
    if result.get("manifest_sha256") != manifest["manifest_sha256"]:
        raise ServingNativeError("SERVING_NATIVE_RESULT_MANIFEST_MISMATCH")
    return result
