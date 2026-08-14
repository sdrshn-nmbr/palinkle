from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path
import statistics
from typing import Any
import urllib.request

from opjax.pallas.laguna_dspark_conformance import canonical_sha256
from opjax.pallas.laguna_dspark_profile import prometheus_values
from opjax.pallas.laguna_speculative import (
    DFLASH,
    DSPARK,
    PLAIN,
    run_replay_benchmark,
    warm_replay_endpoint,
)
from opjax.remote.config import modal_proxy_headers


ENDPOINTS = {
    "plain": (PLAIN, "https://conway--opjax-laguna-speculative-v1-plain.modal.run"),
    **{
        f"dflash-{depth}": (
            DFLASH,
            f"https://conway--opjax-laguna-speculative-v1-trained-dflash{depth}.modal.run",
        )
        for depth in (4, 8, 12, 15)
    },
    **{
        f"dspark-{depth}": (
            DSPARK,
            f"https://conway--opjax-laguna-speculative-v1-trained-dspark{depth}.modal.run",
        )
        for depth in (4, 8, 12, 15)
    },
    "dspark-adaptive": (
        DSPARK,
        "https://conway--opjax-laguna-speculative-v1-trained-dspark-adaptive.modal.run",
    ),
}


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _get(url: str) -> str:
    request = urllib.request.Request(url, headers=modal_proxy_headers())
    with urllib.request.urlopen(request, timeout=1800) as response:
        return response.read().decode()


def _delta(before: str, after: str) -> dict[str, float]:
    baseline = prometheus_values(before)
    return {
        key: value - baseline.get(key, 0.0)
        for key, value in sorted(prometheus_values(after).items())
    }


def _filtered_corpus(
    corpus: dict[str, Any], *, split: str, task_ids: list[str]
) -> dict[str, Any]:
    records = [
        row
        for row in corpus["records"]
        if any(f"--{task_id}--seed-" in row["trajectory"] for task_id in task_ids)
    ]
    trajectories = {row["trajectory"] for row in records}
    if not records:
        raise ValueError(f"LAGUNA_TRAINED_REPLAY_SPLIT_EMPTY:{split}")
    result = {
        "schema_version": 1,
        "kind": "opjax_laguna_trained_replay_split",
        "source_release_sha256": corpus["release_sha256"],
        "split": split,
        "task_ids": task_ids,
        "records": records,
        "counts": {"prompts": len(records), "trajectories": len(trajectories)},
    }
    result["release_sha256"] = canonical_sha256(result)
    return result


def _selection(root: Path, arm: str) -> dict[str, Any]:
    payload = json.loads((root / f"{arm}.json").read_text())
    if payload["arm"] != arm or payload["checkpoint"]["sha256"] == "":
        raise ValueError(f"LAGUNA_TRAINED_SELECTION_INVALID:{arm}")
    return payload


def _cell_summary(result: dict[str, Any], plain: dict[str, Any]) -> dict[str, Any]:
    plain_rows = {row["prompt_id"]: row for row in plain["records"]}
    matches = [
        row
        for row in result["records"]
        if row["completion_token_ids"]
        == plain_rows[row["prompt_id"]]["completion_token_ids"]
    ]
    counters = result["prometheus_delta"]
    drafted = counters.get("vllm:spec_decode_num_draft_tokens_total", 0.0)
    accepted = counters.get("vllm:spec_decode_num_accepted_tokens_total", 0.0)
    rounds = counters.get("vllm:spec_decode_num_drafts_total", 0.0)
    return {
        "requests": len(result["records"]),
        "wall_s": result["wall_s"],
        "output_tps": result["output_tps"],
        "exact_plain_matches": len(matches),
        "exact_match_fraction": len(matches) / len(result["records"]),
        "median_plain_over_cell_latency_on_matches": (
            statistics.median(
                plain_rows[row["prompt_id"]]["elapsed_s"] / row["elapsed_s"]
                for row in matches
            )
            if matches
            else None
        ),
        "speculation": {
            "drafted_tokens": drafted,
            "accepted_tokens": accepted,
            "acceptance_rate": accepted / drafted if drafted else None,
            "accepted_tokens_per_round": accepted / rounds if rounds else None,
        },
        "result_sha256": result["result_sha256"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("calibration", "heldout"), required=True)
    parser.add_argument("--cells", required=True)
    parser.add_argument(
        "--corpus",
        type=Path,
        default=Path("data/pallas/runs/laguna-speculative-v1/replay-corpus.json"),
    )
    parser.add_argument(
        "--corpus-manifest",
        type=Path,
        default=Path("data/pallas/corpora/laguna-speculator-v1/manifest.json"),
    )
    parser.add_argument("--selection-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    selected_cells = [item for item in args.cells.split(",") if item]
    if "plain" not in selected_cells or set(selected_cells) - set(ENDPOINTS):
        raise ValueError("LAGUNA_TRAINED_REPLAY_CELLS_INVALID")
    manifest = json.loads(args.corpus_manifest.read_text())
    corpus = _filtered_corpus(
        json.loads(args.corpus.read_text()),
        split=args.split,
        task_ids=manifest["task_ids"][args.split],
    )
    selections = {arm: _selection(args.selection_root, arm) for arm in (DFLASH, DSPARK)}

    def execute(cell: str) -> tuple[str, dict[str, Any]]:
        arm, endpoint = ENDPOINTS[cell]
        identity = (
            {"plain_target": "poolside/Laguna-XS-2.1"}
            if arm == PLAIN
            else selections[arm]
        )
        _get(f"{endpoint}/health")
        warm_replay_endpoint(
            base_url=endpoint,
            headers=modal_proxy_headers(),
            corpus=corpus,
            max_tokens=8192,
        )
        before = _get(f"{endpoint}/metrics")
        result = run_replay_benchmark(
            arm=arm,
            base_url=endpoint,
            headers=modal_proxy_headers(),
            corpus=corpus,
            concurrency=1,
            max_tokens=8192,
            warmup=False,
            model_identity=identity,
        )
        after = _get(f"{endpoint}/metrics")
        result["cell"] = cell
        result["endpoint"] = endpoint
        result["prometheus_delta"] = _delta(before, after)
        result["result_sha256"] = canonical_sha256(
            {key: value for key, value in result.items() if key != "result_sha256"}
        )
        _write(args.output_root / f"{cell}.json", result)
        return cell, result

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=len(selected_cells)
    ) as executor:
        results = dict(executor.map(execute, selected_cells))
    plain = results["plain"]
    summary = {
        "schema_version": 1,
        "kind": "opjax_laguna_trained_replay_summary",
        "split": args.split,
        "corpus_sha256": corpus["release_sha256"],
        "selection_sha256": {arm: value["sha256"] for arm, value in selections.items()},
        "cells": {
            cell: _cell_summary(result, plain) for cell, result in results.items()
        },
        "files": {
            cell: hashlib.sha256(
                (args.output_root / f"{cell}.json").read_bytes()
            ).hexdigest()
            for cell in results
        },
    }
    summary["sha256"] = canonical_sha256(summary)
    _write(args.output_root / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
