"""Prepare and run the Gate 5 DAPT and DAPT-to-SFT checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import tinker
import torch
from tinker_cookbook.supervised.data import datum_from_model_input_weights
from tinker_cookbook.tokenizer_utils import get_tokenizer

from opjax.pallas.contracts import git_revision
from opjax.pallas.g42_harness import canonical_sha256, file_sha256
from opjax.pallas.g42_training import prepare_g42_training
from opjax.pallas.g5_corpus import validate_g5_dapt_release
from opjax.pallas.phase2_contamination import assert_project_training_content_clean
from opjax.pallas.training import TrainingError, run_prepared_sft


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TrainingError(f"G5_JSON_OBJECT_REQUIRED: {path}")
    return value


def _rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _group_seed(seed: int, split: str, lane: str) -> int:
    digest = hashlib.sha256(f"{split}:{lane}".encode()).hexdigest()
    return seed + int(digest[:8], 16)


def _ordered_unique(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(values))


def _pack_group(
    *,
    rows: list[dict[str, Any]],
    tokenizer: Any,
    max_length: int,
    append_eos: bool,
    seed: int,
    split: str,
    lane: str,
) -> list[dict[str, Any]]:
    ordered = list(rows)
    random.Random(_group_seed(seed, split, lane)).shuffle(ordered)
    tokens: list[int] = []
    origins: list[str] = []
    eos = tokenizer.eos_token_id
    for row in ordered:
        encoded = tokenizer.encode(row["text"], add_special_tokens=False)
        if append_eos:
            encoded = [*encoded, eos]
        if len(encoded) < 2:
            raise TrainingError(f"G5_DAPT_ROW_TOO_SHORT: {row['row_id']}")
        tokens.extend(encoded)
        origins.extend([row["row_id"]] * len(encoded))
    if len(tokens) % max_length == 1 and len(tokens) > max_length:
        split_points = list(range(0, len(tokens) - 2, max_length))
        split_points.append(len(tokens) - 2)
    else:
        split_points = list(range(0, len(tokens), max_length))
    packs = []
    for index, start in enumerate(split_points):
        end = split_points[index + 1] if index + 1 < len(split_points) else len(tokens)
        packed_tokens = tokens[start:end]
        if not 2 <= len(packed_tokens) <= max_length:
            raise TrainingError(
                f"G5_DAPT_PACK_LENGTH_INVALID: {split}:{lane}:{index}:{len(packed_tokens)}"
            )
        packs.append(
            {
                "row_id": f"{split}:{lane}:pack-{index:04d}",
                "split": split,
                "lane": lane,
                "token_count": len(packed_tokens),
                "source_row_ids": _ordered_unique(origins[start:end]),
                "tokens": packed_tokens,
            }
        )
    if sum(pack["token_count"] for pack in packs) != len(tokens):
        raise TrainingError(f"G5_DAPT_TOKEN_CONSERVATION_FAILED: {split}:{lane}")
    return packs


def _pack_rows(
    *, rows: list[dict[str, Any]], tokenizer: Any, packing: Mapping[str, Any]
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["split"], row["lane"])].append(row)
    expected = {
        ("train", "pallas"),
        ("train", "triton"),
        ("validation", "pallas"),
        ("validation", "triton"),
    }
    if set(groups) != expected:
        raise TrainingError(f"G5_DAPT_GROUPS_INVALID: {sorted(groups)}")
    packs = []
    for split, lane in sorted(groups):
        packs.extend(
            _pack_group(
                rows=groups[(split, lane)],
                tokenizer=tokenizer,
                max_length=packing["max_length"],
                append_eos=packing["append_eos"],
                seed=packing["shuffle_seed"],
                split=split,
                lane=lane,
            )
        )
    return packs


def _lane_balanced_order(
    *, packs: list[dict[str, Any]], lane_weights: Mapping[str, float], batch_size: int
) -> list[int]:
    indices_by_lane = {
        lane: [index for index, pack in enumerate(packs) if pack["lane"] == lane]
        for lane in sorted(lane_weights)
    }
    if any(not indices for indices in indices_by_lane.values()):
        raise TrainingError("G5_DAPT_TRAIN_LANE_EMPTY")
    lane_tokens = {
        lane: sum(packs[index]["token_count"] for index in indices)
        for lane, indices in indices_by_lane.items()
    }
    target_total = max(lane_tokens[lane] / lane_weights[lane] for lane in lane_weights)
    schedules: dict[str, list[int]] = {}
    for lane, indices in indices_by_lane.items():
        target = target_total * lane_weights[lane]
        schedule = []
        observed = 0
        for index in itertools.cycle(indices):
            schedule.append(index)
            observed += packs[index]["token_count"]
            if observed >= target:
                break
        schedules[lane] = schedule
    order = []
    lanes = sorted(schedules)
    for position in range(max(len(schedule) for schedule in schedules.values())):
        for lane in lanes:
            if position < len(schedules[lane]):
                order.append(schedules[lane][position])
    if not order:
        raise TrainingError("G5_DAPT_TRAIN_ORDER_EMPTY")
    remainder = len(order) % batch_size
    if remainder:
        order.extend(order[: batch_size - remainder])
    return order


def _datum(tokens: list[int]) -> tinker.Datum:
    model_input = tinker.ModelInput(chunks=[tinker.EncodedTextChunk(tokens=tokens)])
    weights = torch.ones(len(tokens), dtype=torch.float32)
    return datum_from_model_input_weights(
        model_input,
        weights,
        max_length=None,
        reduction="mean",
    )


def _validate_config(config: Mapping[str, Any]) -> None:
    if config.get("schema_version") != 1:
        raise TrainingError("G5_DAPT_CONFIG_SCHEMA_INVALID")
    packing = config.get("packing")
    training = config.get("training")
    if not isinstance(packing, dict) or not isinstance(training, dict):
        raise TrainingError("G5_DAPT_CONFIG_SECTION_INVALID")
    if (
        packing.get("max_length") != 8192
        or packing.get("append_eos") is not True
        or set(packing.get("lane_token_weights", {})) != {"pallas", "triton"}
        or abs(sum(packing["lane_token_weights"].values()) - 1.0) > 1e-12
        or any(value <= 0 for value in packing["lane_token_weights"].values())
    ):
        raise TrainingError("G5_DAPT_PACKING_CONFIG_INVALID")
    required_training = {
        "arm",
        "lora_rank",
        "batch_size",
        "num_epochs",
        "learning_rate",
        "optimizer",
        "loss_fn",
        "training_seed",
        "checkpoint_every_steps",
        "validation_batch_size",
    }
    if not required_training.issubset(training):
        raise TrainingError("G5_DAPT_TRAINING_CONFIG_INVALID")
    if training["num_epochs"] != 1 or training["batch_size"] <= 0:
        raise TrainingError("G5_DAPT_TRAINING_SHAPE_INVALID")


def prepare_g5_dapt(
    *, config_path: Path, corpus_root: Path, repo_root: Path
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    list[tinker.Datum],
    list[int],
    Any,
    dict[str, Any],
]:
    config = _load(config_path)
    _validate_config(config)
    corpus = validate_g5_dapt_release(corpus_root)
    if corpus["release_sha256"] != config["corpus_release_sha256"]:
        raise TrainingError("G5_DAPT_CORPUS_RELEASE_MISMATCH")
    paths = {
        "dataset_sha256": corpus_root / "datasets/dapt.jsonl",
        "train_dataset_sha256": corpus_root / "datasets/train.jsonl",
        "validation_dataset_sha256": corpus_root / "datasets/validation.jsonl",
    }
    for key, path in paths.items():
        if file_sha256(path) != config[key]:
            raise TrainingError(f"G5_DAPT_DATASET_HASH_MISMATCH: {key}")
    tokenizer = get_tokenizer(config["base_model"])
    raw_rows = _rows(corpus_root / "datasets/dapt.jsonl")
    assert_project_training_content_clean(raw_rows)
    packs = _pack_rows(rows=raw_rows, tokenizer=tokenizer, packing=config["packing"])
    train_rows = [pack for pack in packs if pack["split"] == "train"]
    validation_rows = [pack for pack in packs if pack["split"] == "validation"]
    train_datums = [_datum(pack["tokens"]) for pack in train_rows]
    validation_datums = [_datum(pack["tokens"]) for pack in validation_rows]
    order = _lane_balanced_order(
        packs=train_rows,
        lane_weights=config["packing"]["lane_token_weights"],
        batch_size=config["training"]["batch_size"],
    )
    unique_tokens: dict[str, dict[str, int]] = {}
    for split in ("train", "validation"):
        unique_tokens[split] = {
            lane: sum(
                pack["token_count"]
                for pack in packs
                if pack["split"] == split and pack["lane"] == lane
            )
            for lane in ("pallas", "triton")
        }
    effective_by_lane = dict(
        sorted(
            Counter(
                {
                    lane: sum(
                        train_rows[index]["token_count"]
                        for index in order
                        if train_rows[index]["lane"] == lane
                    )
                    for lane in ("pallas", "triton")
                }
            ).items()
        )
    )
    training = {
        **config["training"],
        "max_length": config["packing"]["max_length"],
        "packing": config["packing"],
    }
    preparation: dict[str, Any] = {
        "schema_version": 1,
        "kind": "pallas_g5_dapt_preparation",
        "experiment_id": config["experiment_id"],
        "contract_sha256": canonical_sha256(config),
        "corpus_release_sha256": corpus["release_sha256"],
        "dataset_sha256": config["dataset_sha256"],
        "base_model": config["base_model"],
        "training": training,
        "row_ids": [train_rows[index]["row_id"] for index in order],
        "data": {
            "raw_rows": len(raw_rows),
            "raw_rows_by_lane": dict(sorted(Counter(row["lane"] for row in raw_rows).items())),
            "unique_sequences": {
                split: {
                    lane: sum(
                        pack["split"] == split and pack["lane"] == lane for pack in packs
                    )
                    for lane in ("pallas", "triton")
                }
                for split in ("train", "validation")
            },
            "unique_tokens": unique_tokens,
            "effective_train_sequences": len(order),
            "effective_train_tokens": sum(train_rows[index]["token_count"] for index in order),
            "effective_train_tokens_by_lane": effective_by_lane,
            "supervised_tokens": sum(train_rows[index]["token_count"] - 1 for index in order),
            "minimum_sequence_tokens": min(pack["token_count"] for pack in packs),
            "maximum_sequence_tokens": max(pack["token_count"] for pack in packs),
            "truncated_tokens": 0,
        },
        "opjax_revision": git_revision(repo_root),
    }
    preparation["sha256"] = canonical_sha256(preparation)
    validation = {
        "rows": validation_rows,
        "datums": validation_datums,
        "batch_size": config["training"]["validation_batch_size"],
    }
    return preparation, train_rows, train_datums, order, tokenizer, validation


def train_g5_dapt(
    *,
    config_path: Path,
    corpus_root: Path,
    repo_root: Path,
    out_dir: Path,
    dry_run: bool,
) -> dict[str, Any]:
    preparation, rows, datums, order, tokenizer, validation = prepare_g5_dapt(
        config_path=config_path,
        corpus_root=corpus_root,
        repo_root=repo_root,
    )
    if dry_run:
        return {"ok": True, "dry_run": True, "preparation": preparation}
    return run_prepared_sft(
        preparation=preparation,
        rows=rows,
        datums=datums,
        order=order,
        tokenizer=tokenizer,
        repo_root=repo_root,
        out_dir=out_dir,
        run_kind="pallas_g5_dapt_run",
        gate="G5",
        log_prefix="PALLAS_G5_DAPT_STEP",
        validation=validation,
    )


def _completed_parent_state(path: Path) -> tuple[str, str]:
    manifest = _load(path)
    state = manifest.get("final_state")
    state_path = state.get("path") if isinstance(state, dict) else None
    run_sha256 = manifest.get("run_sha256")
    if (
        manifest.get("status") != "completed"
        or manifest.get("kind") != "pallas_g5_dapt_run"
        or not isinstance(state_path, str)
        or not state_path.startswith("tinker://")
        or not isinstance(run_sha256, str)
    ):
        raise TrainingError("G5_DAPT_PARENT_INVALID")
    return state_path, run_sha256


def validate_g5_training_run(root: Path, *, expected_kind: str) -> dict[str, Any]:
    manifest = _load(root / "manifest.json")
    payload = dict(manifest)
    expected_run_sha256 = payload.pop("run_sha256", None)
    if (
        manifest.get("status") != "completed"
        or manifest.get("kind") != expected_kind
        or canonical_sha256(payload) != expected_run_sha256
        or manifest.get("completed_steps") != manifest.get("total_steps")
    ):
        raise TrainingError(f"G5_TRAINING_RUN_INVALID: {expected_kind}")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict) or set(artifacts) != {
        "events.jsonl",
        "preparation.json",
        "validation.json",
    }:
        raise TrainingError("G5_TRAINING_ARTIFACT_SET_INVALID")
    for relative, expected_hash in artifacts.items():
        path = (root / relative).resolve()
        if (
            not path.is_relative_to(root.resolve())
            or not path.is_file()
            or file_sha256(path) != expected_hash
        ):
            raise TrainingError(f"G5_TRAINING_ARTIFACT_HASH_MISMATCH: {relative}")
    preparation = _load(root / "preparation.json")
    if (
        preparation.get("sha256") != manifest.get("preparation_sha256")
        or preparation.get("corpus_release_sha256")
        != manifest.get("corpus_release_sha256")
        or preparation.get("dataset_sha256") != manifest.get("dataset_sha256")
    ):
        raise TrainingError("G5_TRAINING_PREPARATION_MISMATCH")
    events = _rows(root / "events.jsonl")
    if (
        len(events) != manifest["completed_steps"]
        or [event.get("step") for event in events]
        != list(range(1, manifest["completed_steps"] + 1))
        or any(
            not isinstance(event.get("train_mean_nll"), (int, float))
            or event["train_mean_nll"] < 0
            for event in events
        )
    ):
        raise TrainingError("G5_TRAINING_EVENTS_INVALID")
    validation = _load(root / "validation.json")
    if validation != manifest.get("validation") or any(
        not isinstance(validation.get(stage, {}).get("mean_nll"), (int, float))
        for stage in ("before", "after")
    ):
        raise TrainingError("G5_TRAINING_VALIDATION_INVALID")
    return {
        "ok": True,
        "kind": expected_kind,
        "run_sha256": expected_run_sha256,
        "completed_steps": manifest["completed_steps"],
        "validation": validation,
    }


def train_g5_s1(
    *,
    g5_config_path: Path,
    g5_corpus_root: Path,
    g42_config_path: Path,
    trace_root: Path,
    d0_manifest_path: Path,
    repo_root: Path,
    out_dir: Path,
    dry_run: bool,
) -> dict[str, Any]:
    state_path, parent_run_sha256 = _completed_parent_state(d0_manifest_path)
    preparation, rows, datums, order, tokenizer = prepare_g42_training(
        config_path=g42_config_path,
        trace_root=trace_root,
        repo_root=repo_root,
    )
    _, _, _, _, _, validation = prepare_g5_dapt(
        config_path=g5_config_path,
        corpus_root=g5_corpus_root,
        repo_root=repo_root,
    )
    if dry_run:
        return {
            "ok": True,
            "dry_run": True,
            "preparation": preparation,
            "initial_state_path": state_path,
            "parent_run_sha256": parent_run_sha256,
        }
    return run_prepared_sft(
        preparation=preparation,
        rows=rows,
        datums=datums,
        order=order,
        tokenizer=tokenizer,
        repo_root=repo_root,
        out_dir=out_dir,
        initial_state_path=state_path,
        parent_run_sha256=parent_run_sha256,
        run_kind="pallas_g5_s1_run",
        gate="G5",
        log_prefix="PALLAS_G5_S1_STEP",
        validation=validation,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="opjax-pallas-g5-train")
    commands = parser.add_subparsers(dest="command", required=True)
    dapt = commands.add_parser("dapt")
    dapt.add_argument("--config", type=Path, default=Path("config/pallas/g5-dapt.json"))
    dapt.add_argument("--corpus-root", type=Path, required=True)
    dapt.add_argument("--repo-root", type=Path, default=Path("."))
    dapt.add_argument("--out-dir", type=Path, required=True)
    dapt.add_argument("--dry-run", action="store_true")
    s1 = commands.add_parser("s1")
    s1.add_argument("--g5-config", type=Path, default=Path("config/pallas/g5-dapt.json"))
    s1.add_argument("--g5-corpus-root", type=Path, required=True)
    s1.add_argument("--g42-config", type=Path, default=Path("config/pallas/g42-training.json"))
    s1.add_argument("--trace-root", type=Path, required=True)
    s1.add_argument("--d0-manifest", type=Path, required=True)
    s1.add_argument("--repo-root", type=Path, default=Path("."))
    s1.add_argument("--out-dir", type=Path, required=True)
    s1.add_argument("--dry-run", action="store_true")
    validate = commands.add_parser("validate")
    validate.add_argument("--root", type=Path, required=True)
    validate.add_argument(
        "--kind",
        choices=("pallas_g5_dapt_run", "pallas_g5_s1_run"),
        required=True,
    )
    args = parser.parse_args(argv)
    try:
        if args.command == "dapt":
            result = train_g5_dapt(
                config_path=args.config,
                corpus_root=args.corpus_root,
                repo_root=args.repo_root,
                out_dir=args.out_dir,
                dry_run=args.dry_run,
            )
        elif args.command == "s1":
            result = train_g5_s1(
                g5_config_path=args.g5_config,
                g5_corpus_root=args.g5_corpus_root,
                g42_config_path=args.g42_config,
                trace_root=args.trace_root,
                d0_manifest_path=args.d0_manifest,
                repo_root=args.repo_root,
                out_dir=args.out_dir,
                dry_run=args.dry_run,
            )
        else:
            result = validate_g5_training_run(args.root, expected_kind=args.kind)
    except (TrainingError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"G5_TRAINING_ERROR {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
