from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from opjax.pallas.laguna_serving_native import (
    ServingNativeError,
    captured_prompt_token_ids,
    reconstruct_committed_sample,
    reconstruct_fixed_sample,
    select_final_requests,
    serving_prefix_ends,
    validate_sample,
    write_sample,
)
from opjax.pallas.laguna_dspark_conformance import canonical_sha256


def _write_round(
    root: Path,
    *,
    round_id: int,
    token_ids: list[int],
    positions: list[int],
    feature_value: float,
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    feature_bits = np.asarray([feature_value], dtype=np.float32).view(np.uint32)[0] >> 16
    values = {
        "target_input_ids": np.asarray(token_ids, dtype=np.int32),
        "target_positions": np.asarray(positions, dtype=np.int64),
        "target_aux_hidden_states": np.full(
            (len(token_ids), 10_240), feature_bits, dtype=np.uint16
        ),
        "target_last_hidden_states": np.full(
            (len(token_ids), 2_048), feature_bits, dtype=np.uint16
        ),
    }
    source_dtypes = {
        "target_input_ids": "torch.int32",
        "target_positions": "torch.int64",
        "target_aux_hidden_states": "torch.bfloat16",
        "target_last_hidden_states": "torch.bfloat16",
    }
    with (root / "target-ledger.jsonl").open("a", encoding="utf-8") as ledger:
        for name, value in values.items():
            path = root / f"{name}-{round_id:03d}.npy"
            np.save(path, value, allow_pickle=False)
            record = {
                "name": name,
                "index": round_id,
                "path": path.name,
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "source_dtype": source_dtypes[name],
                "round": round_id,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            ledger.write(json.dumps(record, sort_keys=True) + "\n")


def test_select_final_requests_preserves_trajectory_splits() -> None:
    corpus = {
        "records": [
            {"trajectory": "m--train-task--seed-0", "call": 1},
            {"trajectory": "m--train-task--seed-0", "call": 2},
            {"trajectory": "m--cal-task--seed-0", "call": 1},
            {"trajectory": "m--held-task--seed-0", "call": 1},
        ]
    }
    manifest = {
        "task_ids": {
            "train": ["train-task"],
            "calibration": ["cal-task"],
            "heldout": ["held-task"],
        },
        "trajectories": {"train": 1, "calibration": 1, "heldout": 1},
    }
    selected = select_final_requests(corpus, manifest)
    assert selected["train"][0]["call"] == 2
    assert {key: len(value) for key, value in selected.items()} == {
        "train": 1,
        "calibration": 1,
        "heldout": 1,
    }


def test_reconstruct_committed_sample_ignores_rejected_drafts_and_overwrites(
    tmp_path: Path,
) -> None:
    prompt = [10, 11]
    completion = list(range(20, 36))
    committed = prompt + completion
    _write_round(
        tmp_path,
        round_id=0,
        token_ids=committed[:10] + [999],
        positions=list(range(10)) + [10],
        feature_value=1.0,
    )
    _write_round(
        tmp_path,
        round_id=1,
        token_ids=committed[8:],
        positions=list(range(8, len(committed))),
        feature_value=2.0,
    )
    sample, metadata = reconstruct_committed_sample(
        session_root=tmp_path,
        prompt_token_ids=prompt,
        completion_token_ids=completion,
    )
    assert sample["input_ids"].tolist() == committed
    assert sample["loss_mask"].tolist() == [0, 0] + [1] * 16
    assert sample["target_hidden_states"].dtype == np.uint16
    expected_one = np.asarray([1.0], dtype=np.float32).view(np.uint32)[0] >> 16
    expected_two = np.asarray([2.0], dtype=np.float32).view(np.uint32)[0] >> 16
    assert np.all(sample["target_hidden_states"][:8] == expected_one)
    assert np.all(sample["target_hidden_states"][8:] == expected_two)
    assert metadata["rejected_or_out_of_range_rows"] == 1
    assert metadata["source_round_max"] == 1


def test_reconstruct_rejects_missing_committed_position(tmp_path: Path) -> None:
    _write_round(
        tmp_path,
        round_id=0,
        token_ids=[1, 2] + list(range(10, 24)),
        positions=list(range(16)),
        feature_value=1.0,
    )
    with pytest.raises(ServingNativeError, match="COMMITTED_COVERAGE_MISSING"):
        reconstruct_committed_sample(
            session_root=tmp_path,
            prompt_token_ids=[1, 2],
            completion_token_ids=list(range(10, 25)),
        )


def test_captured_prompt_uses_runtime_tokens(tmp_path: Path) -> None:
    _write_round(
        tmp_path,
        round_id=0,
        token_ids=[8, 9, 10, 99],
        positions=[0, 1, 2, 3],
        feature_value=1.0,
    )
    assert captured_prompt_token_ids(
        session_root=tmp_path, prompt_token_count=3
    ) == [8, 9, 10]


def test_write_sample_binds_every_array(tmp_path: Path) -> None:
    sample = {
        "input_ids": np.arange(16, dtype=np.int32),
        "attention_mask": np.ones(16, dtype=np.uint8),
        "loss_mask": np.ones(16, dtype=np.uint8),
        "target_hidden_states": np.zeros((16, 10_240), dtype=np.uint16),
        "target_last_hidden_states": np.zeros((16, 2_048), dtype=np.uint16),
    }
    metadata = {"metadata_sha256": "x"}
    manifest = write_sample(tmp_path / "sample", sample=sample, metadata=metadata)
    assert set(manifest["files"]) == set(sample)
    for item in manifest["files"].values():
        path = tmp_path / "sample" / item["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"]


def test_validate_sample_rejects_mutation(tmp_path: Path) -> None:
    sample = {
        "input_ids": np.arange(16, dtype=np.int32),
        "attention_mask": np.ones(16, dtype=np.uint8),
        "loss_mask": np.ones(16, dtype=np.uint8),
        "target_hidden_states": np.zeros((16, 10_240), dtype=np.uint16),
        "target_last_hidden_states": np.zeros((16, 2_048), dtype=np.uint16),
    }
    root = tmp_path / "sample"
    manifest = write_sample(
        root,
        sample=sample,
        metadata={"tokens": 16, "metadata_sha256": "x"},
    )
    result = {
        "manifest_sha256": manifest["manifest_sha256"],
        "prompt_id": "p",
    }
    result["result_sha256"] = canonical_sha256(result)
    (root / "capture-result.json").write_text(json.dumps(result), encoding="utf-8")
    assert validate_sample(root)["prompt_id"] == "p"
    with (root / "input_ids.npy").open("r+b") as handle:
        handle.seek(-1, 2)
        handle.write(b"x")
    with pytest.raises(ServingNativeError, match="SAMPLE_FILE_INVALID"):
        validate_sample(root)


def test_serving_prefix_ends_preserves_turns_and_chunks_assistant() -> None:
    mask = np.asarray([0] * 5 + [1] * 32 + [0] * 3 + [1] * 4, dtype=np.uint8)
    assert serving_prefix_ends(mask, proposal_length=15) == [5, 20, 35, 37, 40, 44]


def test_reconstruct_fixed_sample_uses_only_matching_runtime_rows(
    tmp_path: Path,
) -> None:
    tokens = np.arange(32, dtype=np.int32)
    _write_round(
        tmp_path,
        round_id=0,
        token_ids=[*tokens[:20], 999],
        positions=[*range(20), 20],
        feature_value=1.0,
    )
    _write_round(
        tmp_path,
        round_id=1,
        token_ids=tokens[16:].tolist(),
        positions=list(range(16, 32)),
        feature_value=2.0,
    )
    sample, metadata = reconstruct_fixed_sample(
        session_root=tmp_path,
        input_ids=tokens,
        loss_mask=np.ones(32, dtype=np.uint8),
    )
    assert sample["input_ids"].tolist() == tokens.tolist()
    assert metadata["mode"] == "fixed_arm_independent_prefix_replay"
    assert metadata["rejected_or_out_of_range_rows"] == 1
    assert metadata["rounds"][1]["committed_rows"] == 16
    assert metadata["feature_selection_policy"] == "first_causal_observation_wins"
    assert metadata["overlap_rows"] == 4
    assert metadata["divergent_overlap_rows"] == 4
    expected_one = np.asarray([1.0], dtype=np.float32).view(np.uint32)[0] >> 16
    assert sample["target_hidden_states"][9, 0] == expected_one


def test_reconstruct_rejects_coincidental_match_after_first_mismatch(
    tmp_path: Path,
) -> None:
    tokens = np.arange(16, dtype=np.int32)
    _write_round(
        tmp_path,
        round_id=0,
        token_ids=[*tokens[:8], 999, int(tokens[9])],
        positions=list(range(10)),
        feature_value=1.0,
    )
    _write_round(
        tmp_path,
        round_id=1,
        token_ids=tokens[8:].tolist(),
        positions=list(range(8, 16)),
        feature_value=2.0,
    )
    sample, metadata = reconstruct_fixed_sample(
        session_root=tmp_path,
        input_ids=tokens,
        loss_mask=np.ones(16, dtype=np.uint8),
    )
    expected_two = np.asarray([2.0], dtype=np.float32).view(np.uint32)[0] >> 16
    assert sample["target_hidden_states"][9, 0] == expected_two
    assert metadata["rounds"][0]["committed_rows"] == 8
    assert metadata["rejected_or_out_of_range_rows"] == 2
