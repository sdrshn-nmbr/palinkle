from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path
import random
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


CELLS = {
    "plain": (PLAIN, "https://conway--opjax-laguna-speculative-v1-plain.modal.run"),
    "dflash-15": (
        DFLASH,
        "https://conway--opjax-laguna-speculative-v1-dflash.modal.run",
    ),
    **{
        f"dspark-{depth}": (
            DSPARK,
            f"https://conway--opjax-laguna-speculative-v1-dspark{depth}.modal.run",
        )
        for depth in (4, 8, 12, 15)
    },
}


def _write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _get(url: str) -> str:
    request = urllib.request.Request(url, headers=modal_proxy_headers())
    with urllib.request.urlopen(request, timeout=1800) as response:
        return response.read().decode()


def _delta(before: str, after: str) -> dict[str, float]:
    before_values = prometheus_values(before)
    return {
        key: value - before_values.get(key, 0.0)
        for key, value in sorted(prometheus_values(after).items())
    }


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * fraction)]


def _distribution(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.mean(values),
        "p50": _percentile(values, 0.5),
        "p95": _percentile(values, 0.95),
        "p99": _percentile(values, 0.99),
    }


def _cluster_bootstrap_latency_ratio(
    *,
    plain_rows: dict[str, dict[str, Any]],
    arm_rows: dict[str, dict[str, Any]],
    prompt_ids: list[str],
    samples: int = 10_000,
) -> dict[str, float]:
    grouped: dict[str, list[str]] = {}
    for prompt_id in prompt_ids:
        grouped.setdefault(plain_rows[prompt_id]["trajectory"], []).append(prompt_id)
    trajectories = sorted(grouped)
    if not trajectories:
        return {"clusters": 0.0, "samples": float(samples)}

    def ratio(selected: list[str]) -> float:
        selected_ids = [prompt_id for name in selected for prompt_id in grouped[name]]
        return sum(
            plain_rows[prompt_id]["elapsed_s"] for prompt_id in selected_ids
        ) / sum(arm_rows[prompt_id]["elapsed_s"] for prompt_id in selected_ids)

    rng = random.Random(0)
    draws = sorted(
        ratio([rng.choice(trajectories) for _ in trajectories]) for _ in range(samples)
    )
    return {
        "clusters": float(len(trajectories)),
        "samples": float(samples),
        "plain_over_cell_point_estimate": ratio(trajectories),
        "ci95_low": _percentile(draws, 0.025),
        "ci95_high": _percentile(draws, 0.975),
    }


def _summary(root: Path, corpus: dict[str, Any]) -> dict[str, Any]:
    cells = {
        cell: json.loads((root / f"{cell}.json").read_text(encoding="utf-8"))
        for cell in CELLS
    }
    plain_rows = {row["prompt_id"]: row for row in cells["plain"]["records"]}
    measurements: dict[str, Any] = {}
    for cell, result in cells.items():
        rows = result["records"]
        counters = result["prometheus_delta"]
        drafted = counters.get("vllm:spec_decode_num_draft_tokens_total", 0.0)
        accepted = counters.get("vllm:spec_decode_num_accepted_tokens_total", 0.0)
        rounds = counters.get("vllm:spec_decode_num_drafts_total", 0.0)
        matches = [
            row
            for row in rows
            if row["completion_token_ids"]
            == plain_rows[row["prompt_id"]]["completion_token_ids"]
        ]
        arm_rows = {row["prompt_id"]: row for row in rows}
        match_ids = [row["prompt_id"] for row in matches]
        measurements[cell] = {
            "requests": len(rows),
            "wall_s": result["wall_s"],
            "request_throughput": len(rows) / result["wall_s"],
            "completion_tokens": result["completion_tokens"],
            "output_tps": result["output_tps"],
            "client_latency_s": _distribution(
                [float(row["elapsed_s"]) for row in rows]
            ),
            "finish_reasons": {
                reason: sum(row["finish_reason"] == reason for row in rows)
                for reason in sorted({row["finish_reason"] for row in rows})
            },
            "exact_plain_matches": len(matches),
            "exact_plain_match_fraction": len(matches) / len(rows),
            "exact_match_plain_over_cell_latency": (
                None
                if cell == "plain" or not matches
                else _distribution(
                    [
                        plain_rows[row["prompt_id"]]["elapsed_s"] / row["elapsed_s"]
                        for row in matches
                    ]
                )
            ),
            "cluster_bootstrap_all_request_latency": (
                None
                if cell == "plain"
                else _cluster_bootstrap_latency_ratio(
                    plain_rows=plain_rows,
                    arm_rows=arm_rows,
                    prompt_ids=sorted(plain_rows),
                )
            ),
            "cluster_bootstrap_exact_match_latency": (
                None
                if cell == "plain"
                else _cluster_bootstrap_latency_ratio(
                    plain_rows=plain_rows,
                    arm_rows=arm_rows,
                    prompt_ids=match_ids,
                )
            ),
            "speculation": {
                "draft_rounds": rounds,
                "drafted_tokens": drafted,
                "accepted_tokens": accepted,
                "acceptance_rate": accepted / drafted if drafted else None,
                "accepted_tokens_per_round": accepted / rounds if rounds else None,
                "output_tokens_per_target_verification": (
                    result["completion_tokens"] / rounds if rounds else None
                ),
                "accepted_by_position": {
                    key.rsplit(".", maxsplit=1)[-1]: value
                    for key, value in counters.items()
                    if "accepted_tokens_per_pos" in key
                },
            },
            "result_sha256": result["result_sha256"],
            "file_sha256": hashlib.sha256(
                (root / f"{cell}.json").read_bytes()
            ).hexdigest(),
        }
    summary: dict[str, Any] = {
        "schema_version": 1,
        "kind": "opjax_laguna_dspark_rope_corrected_replay",
        "corpus_sha256": corpus["release_sha256"],
        "prompts": corpus["counts"]["prompts"],
        "trajectories": corpus["counts"]["trajectories"],
        "sampling": {"temperature": 0.0, "seed": 0, "max_tokens": 8192},
        "cells": measurements,
        "claim_boundary": (
            "matched historical prompts; divergent output is operational behavior, "
            "not a change in target weights"
        ),
    }
    summary["summary_sha256"] = canonical_sha256(summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--corpus",
        type=Path,
        default=Path("data/pallas/runs/laguna-speculative-v1/replay-corpus.json"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/pallas/runs/laguna-dspark-corrected-v1/replay"),
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--cells",
        default=",".join(CELLS),
        help="Comma-separated cells to execute. Summary is written only when all cells exist.",
    )
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    corpus = json.loads(args.corpus.read_text(encoding="utf-8"))
    if args.summarize_only:
        summary = _summary(args.output_root, corpus)
        _write(args.output_root / "summary.json", summary)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return
    args.output_root.mkdir(parents=True, exist_ok=True)
    selected_cells = tuple(value for value in args.cells.split(",") if value)
    unknown = sorted(set(selected_cells) - set(CELLS))
    if unknown:
        raise ValueError(f"LAGUNA_REPLAY_CELLS_INVALID:{','.join(unknown)}")

    def run_cell(item: tuple[str, tuple[str, str]]) -> tuple[str, float]:
        cell, (arm, endpoint) = item
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
            limit=args.limit,
            warmup=False,
        )
        after = _get(f"{endpoint}/metrics")
        result["cell"] = cell
        result["endpoint"] = endpoint
        result["prometheus_delta"] = _delta(before, after)
        result["result_sha256"] = canonical_sha256(
            {key: value for key, value in result.items() if key != "result_sha256"}
        )
        _write(args.output_root / f"{cell}.json", result)
        return cell, result["output_tps"]

    selected = [(cell, CELLS[cell]) for cell in selected_cells]
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(selected)) as executor:
        for cell, output_tps in executor.map(run_cell, selected):
            print(f"CELL={cell} OUTPUT_TPS={output_tps:.6f}", flush=True)
    missing = [
        cell for cell in CELLS if not (args.output_root / f"{cell}.json").is_file()
    ]
    if not missing:
        summary = _summary(args.output_root, corpus)
        _write(args.output_root / "summary.json", summary)
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print(f"LAGUNA_REPLAY_PENDING_CELLS={','.join(missing)}", flush=True)


if __name__ == "__main__":
    main()
