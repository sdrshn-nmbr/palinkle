from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import json
import random
import re
import statistics
import urllib.request
from pathlib import Path
from typing import Any

from opjax.pallas.laguna_speculative import (
    ARMS,
    build_replay_corpus,
    canonical_sha256,
    run_replay_benchmark,
    validate_model_manifest,
    warm_replay_endpoint,
)
from opjax.remote.config import modal_proxy_headers

ENDPOINTS = {
    arm: f"https://conway--opjax-laguna-speculative-v1-{arm}.modal.run"
    for arm in ARMS
}
DEPLOYED_SOURCE_FILES = (
    Path("src/opjax/pallas/laguna_speculative.py"),
    Path("src/opjax/remote/laguna_dspark_vllm_model.py"),
    Path("src/opjax/remote/laguna_speculative_vllm.py"),
    Path("src/opjax/remote/laguna_vllm_entrypoint.py"),
)


def _write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _get(url: str) -> str:
    request = urllib.request.Request(url, headers=modal_proxy_headers())
    with urllib.request.urlopen(request, timeout=1800) as response:
        return response.read().decode()


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * quantile)]


def _distribution(values: list[float]) -> dict[str, float]:
    if not values:
        return {"count": 0.0}
    return {
        "count": float(len(values)),
        "mean": statistics.mean(values),
        "p50": _percentile(values, 0.50),
        "p95": _percentile(values, 0.95),
        "p99": _percentile(values, 0.99),
    }


def _metric_summary(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    keys = (
        "generation_time_ms",
        "mean_itl_ms",
        "queue_time_ms",
        "time_to_first_token_ms",
        "tokens_per_second",
    )
    result: dict[str, dict[str, float]] = {}
    for key in keys:
        values = [float(row["server_metrics"][key]) for row in rows]
        result[key] = _distribution(values)
    return result


_COUNTER_PATTERN = re.compile(
    r'^(?P<name>vllm:[^{ ]+)(?:\{(?P<labels>[^}]*)\})? (?P<value>[-+0-9.eE]+)$'
)


def _prometheus_values(text: str) -> dict[str, float]:
    selected = {
        "vllm:spec_decode_num_drafts_total",
        "vllm:spec_decode_num_draft_tokens_total",
        "vllm:spec_decode_num_accepted_tokens_total",
        "vllm:spec_decode_num_accepted_tokens_per_pos_total",
        "vllm:prefix_cache_queries_total",
        "vllm:prefix_cache_hits_total",
        "vllm:num_preemptions_total",
        "vllm:request_success_total",
        "vllm:request_aborted_total",
        "vllm:generation_tokens_total",
        "vllm:prompt_tokens_total",
    }
    values: dict[str, float] = {}
    for line in text.splitlines():
        match = _COUNTER_PATTERN.match(line)
        if match is None or match.group("name") not in selected:
            continue
        name = match.group("name")
        labels = match.group("labels") or ""
        position = re.search(r'position="(?P<position>\d+)"', labels)
        if position:
            name = f"{name}.position_{position.group('position')}"
        finished_reason = re.search(r'finished_reason="(?P<reason>[^"]+)"', labels)
        if finished_reason:
            name = f"{name}.finished_{finished_reason.group('reason')}"
        values[name] = float(match.group("value"))
    return values


def _prometheus_delta(before: str, after: str) -> dict[str, float]:
    before_values = _prometheus_values(before)
    after_values = _prometheus_values(after)
    return {
        key: value - before_values.get(key, 0.0)
        for key, value in sorted(after_values.items())
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
        return sum(plain_rows[prompt_id]["elapsed_s"] for prompt_id in selected_ids) / sum(
            arm_rows[prompt_id]["elapsed_s"] for prompt_id in selected_ids
        )

    rng = random.Random(0)
    draws = sorted(
        ratio([rng.choice(trajectories) for _ in trajectories])
        for _ in range(samples)
    )
    return {
        "clusters": float(len(trajectories)),
        "samples": float(samples),
        "plain_over_arm_point_estimate": ratio(trajectories),
        "ci95_low": _percentile(draws, 0.025),
        "ci95_high": _percentile(draws, 0.975),
    }


def _runtime_artifact_summary(root: Path, arm: str) -> dict[str, Any]:
    artifact_root = root / "runtime-files" / arm
    runtime_path = artifact_root / "runtime.json"
    gpu_path = artifact_root / "gpu.csv"
    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    expected_runtime_sha = runtime.pop("sha256")
    if canonical_sha256(runtime) != expected_runtime_sha:
        raise ValueError(f"LAGUNA_RUNTIME_FINGERPRINT_HASH_INVALID:{arm}")
    with gpu_path.open(encoding="utf-8", newline="") as source:
        rows = [
            {key.strip(): value.strip() for key, value in row.items()}
            for row in csv.DictReader(source)
        ]
    if not rows:
        raise ValueError(f"LAGUNA_GPU_TELEMETRY_EMPTY:{arm}")

    def numeric(column: str, suffix: str) -> list[float]:
        return [float(row[column].removesuffix(suffix).strip()) for row in rows]

    return {
        "runtime_fingerprint_sha256": expected_runtime_sha,
        "runtime_file_sha256": hashlib.sha256(runtime_path.read_bytes()).hexdigest(),
        "gpu_file_sha256": hashlib.sha256(gpu_path.read_bytes()).hexdigest(),
        "vllm_observed_build": runtime["vllm_observed_build"],
        "image": runtime["image"],
        "argv_sha256": canonical_sha256(runtime["argv"]),
        "gpu_samples": len(rows),
        "gpu_telemetry_scope": "container_lifetime_includes_startup_and_post_run_idle",
        "gpu_memory_used_mib": _distribution(numeric("memory.used [MiB]", "MiB")),
        "gpu_utilization_percent": _distribution(
            numeric("utilization.gpu [%]", "%")
        ),
        "gpu_power_watts": _distribution(numeric("power.draw [W]", "W")),
    }


def _summarize(root: Path, concurrencies: tuple[int, ...]) -> dict[str, Any]:
    results: dict[tuple[str, int], dict[str, Any]] = {}
    for arm in ARMS:
        for concurrency in concurrencies:
            results[(arm, concurrency)] = json.loads(
                (root / f"{arm}-c{concurrency}.json").read_text(encoding="utf-8")
            )
    parity = {}
    for concurrency in concurrencies:
        token_ids = {
            arm: {
                row["prompt_id"]: row["completion_token_ids"]
                for row in results[(arm, concurrency)]["records"]
            }
            for arm in ARMS
        }
        prompt_ids = sorted(token_ids[ARMS[0]])
        parity[str(concurrency)] = {
            "prompts": len(prompt_ids),
            "all_three_exact_matches": sum(
                len({tuple(token_ids[arm][prompt_id]) for arm in ARMS}) == 1
                for prompt_id in prompt_ids
            ),
            "plain_dflash_exact_matches": sum(
                token_ids["plain"][prompt_id] == token_ids["dflash"][prompt_id]
                for prompt_id in prompt_ids
            ),
            "plain_dspark_exact_matches": sum(
                token_ids["plain"][prompt_id] == token_ids["dspark"][prompt_id]
                for prompt_id in prompt_ids
            ),
        }
    measurements = {}
    for concurrency in concurrencies:
        plain_tps = results[("plain", concurrency)]["output_tps"]
        for arm in ARMS:
            result = results[(arm, concurrency)]
            rows = result["records"]
            finish_reasons = sorted({row["finish_reason"] for row in rows})
            measurements[f"{arm}-c{concurrency}"] = {
                "wall_s": result["wall_s"],
                "request_throughput": len(rows) / result["wall_s"],
                "completion_tokens": result["completion_tokens"],
                "output_tps": result["output_tps"],
                "speedup_over_plain": result["output_tps"] / plain_tps,
                "client_latency_s": result["request_latency_s"],
                "completion_length": _distribution(
                    [float(row["completion_tokens"]) for row in rows]
                ),
                "finish_reasons": {
                    reason: sum(row["finish_reason"] == reason for row in rows)
                    for reason in finish_reasons
                },
                "server_metrics": _metric_summary(rows),
                "prometheus_delta": _prometheus_delta(
                    result["prometheus_before"], result["prometheus_after"]
                ),
            }
    paired_exact_matches: dict[str, Any] = {}
    for concurrency in concurrencies:
        plain_rows = {
            row["prompt_id"]: row
            for row in results[("plain", concurrency)]["records"]
        }
        for arm in ("dflash", "dspark"):
            arm_rows = {
                row["prompt_id"]: row
                for row in results[(arm, concurrency)]["records"]
            }
            matched_ids = [
                prompt_id
                for prompt_id, plain_row in plain_rows.items()
                if plain_row["completion_token_ids"]
                == arm_rows[prompt_id]["completion_token_ids"]
            ]
            unmatched_ids = sorted(set(plain_rows) - set(matched_ids))
            paired_exact_matches[f"{arm}-c{concurrency}"] = {
                "matches": len(matched_ids),
                "plain_over_arm_client_latency_speedup": _distribution(
                    [
                        plain_rows[prompt_id]["elapsed_s"]
                        / arm_rows[prompt_id]["elapsed_s"]
                        for prompt_id in matched_ids
                    ]
                ),
                "plain_over_arm_generation_speedup": _distribution(
                    [
                        plain_rows[prompt_id]["server_metrics"][
                            "generation_time_ms"
                        ]
                        / arm_rows[prompt_id]["server_metrics"][
                            "generation_time_ms"
                        ]
                        for prompt_id in matched_ids
                    ]
                ),
                "matched_historical_completion_tokens": _distribution(
                    [
                        float(
                            plain_rows[prompt_id]["historical_completion_tokens"]
                        )
                        for prompt_id in matched_ids
                    ]
                ),
                "unmatched_historical_completion_tokens": _distribution(
                    [
                        float(
                            plain_rows[prompt_id]["historical_completion_tokens"]
                        )
                        for prompt_id in unmatched_ids
                    ]
                ),
                "cluster_bootstrap_exact_match_latency": (
                    _cluster_bootstrap_latency_ratio(
                        plain_rows=plain_rows,
                        arm_rows=arm_rows,
                        prompt_ids=matched_ids,
                    )
                ),
                "cluster_bootstrap_all_request_latency": (
                    _cluster_bootstrap_latency_ratio(
                        plain_rows=plain_rows,
                        arm_rows=arm_rows,
                        prompt_ids=sorted(plain_rows),
                    )
                ),
            }
    replay_corpus_path = root / "replay-corpus.json"
    replay_corpus = json.loads(replay_corpus_path.read_text(encoding="utf-8"))
    for key, result in results.items():
        expected_result_sha = result["result_sha256"]
        actual_result_sha = canonical_sha256(
            {name: value for name, value in result.items() if name != "result_sha256"}
        )
        if actual_result_sha != expected_result_sha:
            raise ValueError(f"LAGUNA_CELL_RESULT_HASH_INVALID:{key}")
        if result["corpus_sha256"] != replay_corpus["release_sha256"]:
            raise ValueError(f"LAGUNA_CELL_CORPUS_BINDING_INVALID:{key}")
    cell_evidence = {
        f"{arm}-c{concurrency}": {
            "file_sha256": hashlib.sha256(
                (root / f"{arm}-c{concurrency}.json").read_bytes()
            ).hexdigest(),
            "result_sha256": results[(arm, concurrency)]["result_sha256"],
            "corpus_sha256": results[(arm, concurrency)]["corpus_sha256"],
            "records": len(results[(arm, concurrency)]["records"]),
        }
        for concurrency in concurrencies
        for arm in ARMS
    }
    deployed_source = {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in DEPLOYED_SOURCE_FILES
    }
    summary: dict[str, Any] = {
        "schema_version": 1,
        "kind": "opjax_laguna_speculative_replay_summary",
        "model_manifest": validate_model_manifest(),
        "concurrencies": list(concurrencies),
        "parity": parity,
        "measurements": measurements,
        "paired_exact_matches": paired_exact_matches,
        "measurement_contract": {
            "sampling": "greedy_temperature_0_seed_0",
            "max_tokens": results[("plain", concurrencies[0])]["max_tokens"],
            "warmup_excluded_from_prometheus_delta": True,
            "trajectory_order_preserved": True,
            "inputs": "fixed_historical_phase32_prefixes",
        },
        "claim_boundary": (
            "matched_input_serving_test_not_live_agent_rollout; after output "
            "divergence each next prompt still contains the historical assistant output"
        ),
        "runtime_artifacts": {
            arm: _runtime_artifact_summary(root, arm) for arm in ARMS
        },
        "cell_evidence": cell_evidence,
        "replay_corpus": {
            "file_sha256": hashlib.sha256(replay_corpus_path.read_bytes()).hexdigest(),
            "release_sha256": replay_corpus["release_sha256"],
            "prompts": replay_corpus["counts"]["prompts"],
            "trajectories": replay_corpus["counts"]["trajectories"],
        },
        "deployed_source_sha256": deployed_source,
    }
    summary["sha256"] = canonical_sha256(summary)
    return summary


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
        default=Path("data/pallas/runs/laguna-speculative-v1"),
    )
    parser.add_argument("--concurrencies", default="1")
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    concurrencies = tuple(int(value) for value in args.concurrencies.split(","))
    if args.summarize_only:
        summary = _summarize(args.output_root, concurrencies)
        _write(args.output_root / "summary.json", summary)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return
    corpus = build_replay_corpus(sample_root=args.sample_root.resolve())
    _write(args.output_root / "replay-corpus.json", corpus)

    for concurrency in concurrencies:
        def run_arm(arm: str) -> tuple[str, float]:
            _get(f"{ENDPOINTS[arm]}/health")
            warm_replay_endpoint(
                base_url=ENDPOINTS[arm],
                headers=modal_proxy_headers(),
                corpus=corpus,
                max_tokens=args.max_tokens,
            )
            before = _get(f"{ENDPOINTS[arm]}/metrics")
            result = run_replay_benchmark(
                arm=arm,
                base_url=ENDPOINTS[arm],
                headers=modal_proxy_headers(),
                corpus=corpus,
                concurrency=concurrency,
                max_tokens=args.max_tokens,
                warmup=False,
            )
            after = _get(f"{ENDPOINTS[arm]}/metrics")
            result["prometheus_before"] = before
            result["prometheus_after"] = after
            result["result_sha256"] = canonical_sha256(
                {key: value for key, value in result.items() if key != "result_sha256"}
            )
            _write(args.output_root / f"{arm}-c{concurrency}.json", result)
            return arm, result["output_tps"]

        with concurrent.futures.ThreadPoolExecutor(max_workers=len(ARMS)) as executor:
            for arm, output_tps in executor.map(run_arm, ARMS):
                print(
                    f"LAGUNA_SPECULATIVE_RESULT concurrency={concurrency} "
                    f"arm={arm} output_tps={output_tps:.6f}",
                    flush=True,
                )
    summary = _summarize(args.output_root, concurrencies)
    _write(args.output_root / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
