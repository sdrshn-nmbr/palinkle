"""Train one frozen G4.3 learning-curve replicate."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

from tinker_cookbook import model_info, renderers
from tinker_cookbook.supervised.data import conversation_to_datum
from tinker_cookbook.tokenizer_utils import get_tokenizer

from opjax.pallas.contracts import git_revision
from opjax.pallas.g42_harness import canonical_sha256, file_sha256
from opjax.pallas.g43_corpus import validate_trace_subset
from opjax.pallas.training import TrainingError, run_prepared_sft
from opjax.pallas.phase2_contamination import assert_project_training_content_clean


def _rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def prepare_g43_training(
    *, config_path: Path, trace_root: Path, repo_root: Path
) -> tuple[dict[str, Any], list[dict[str, Any]], list[Any], list[int], Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    validation = validate_trace_subset(trace_root)
    required = {
        "schema_version": 1,
        "trace_release_sha256": validation["release_sha256"],
        "verified_trajectories": validation["trajectory_count"],
        "prefix_sft_rows": validation["row_count"],
    }
    for key, value in required.items():
        if config.get(key) != value:
            raise TrainingError(f"G43_TRAINING_CONFIG_MISMATCH: {key}")
    dataset = trace_root / "datasets" / "prefix-sft.jsonl"
    if file_sha256(dataset) != config.get("dataset_sha256"):
        raise TrainingError("G43_DATASET_HASH_MISMATCH")
    rows = _rows(dataset)
    assert_project_training_content_clean(rows)
    if len({row["row_id"] for row in rows}) != len(rows):
        raise TrainingError("G43_TRAINING_ROW_ID_DUPLICATE")
    recommended = model_info.get_recommended_renderer_name(config["base_model"])
    if recommended != config["renderer"]:
        raise TrainingError(
            f"RENDERER_MISMATCH: expected={config['renderer']} observed={recommended}"
        )
    tokenizer = get_tokenizer(config["base_model"])
    renderer = renderers.get_renderer(
        config["renderer"], tokenizer, model_name=config["base_model"]
    )
    train_on = renderers.TrainOnWhat(config["train_on"])
    datums = []
    lengths = []
    supervised_tokens = 0
    for row in rows:
        model_input, weights = renderer.build_supervised_example(
            row["messages"], train_on_what=train_on
        )
        length = model_input.length + 1
        if length > config["max_length"]:
            raise TrainingError(
                f"G43_CONTEXT_EXCEEDED: {row['row_id']}:{length}>{config['max_length']}"
            )
        datums.append(conversation_to_datum(row["messages"], renderer, None, train_on))
        lengths.append(length)
        supervised_tokens += sum(float(weight) > 0 for weight in weights)
    order = list(range(len(rows)))
    random.Random(config["shuffle_seed"]).shuffle(order)
    if len(order) % config["batch_size"]:
        raise TrainingError("G43_SFT_BATCH_REMAINDER")
    preparation: dict[str, Any] = {
        "schema_version": 1,
        "kind": "pallas_g43_sft_preparation",
        "experiment_id": config["experiment_id"],
        "contract_sha256": canonical_sha256(config),
        "corpus_release_sha256": validation["release_sha256"],
        "dataset_sha256": config["dataset_sha256"],
        "base_model": config["base_model"],
        "training": config,
        "row_ids": [rows[index]["row_id"] for index in order],
        "data": {
            "rows": len(rows),
            "trajectories": validation["trajectory_count"],
            "sequence_tokens": sum(lengths),
            "supervised_tokens": supervised_tokens,
            "minimum_sequence_tokens": min(lengths),
            "maximum_sequence_tokens": max(lengths),
            "truncated_rows": 0,
        },
        "opjax_revision": git_revision(repo_root),
    }
    preparation["sha256"] = canonical_sha256(preparation)
    return preparation, rows, datums, order, tokenizer


def train_g43(
    *,
    config_path: Path,
    trace_root: Path,
    repo_root: Path,
    out_dir: Path,
    dry_run: bool,
) -> dict[str, Any]:
    preparation, rows, datums, order, tokenizer = prepare_g43_training(
        config_path=config_path,
        trace_root=trace_root,
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
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="opjax-pallas-g43-train")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--trace-root", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = train_g43(
            config_path=args.config,
            trace_root=args.trace_root,
            repo_root=args.repo_root,
            out_dir=args.out_dir,
            dry_run=args.dry_run,
        )
    except (TrainingError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(
            json.dumps(
                {"ok": False, "error": str(exc), "error_type": type(exc).__name__},
                indent=2,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
