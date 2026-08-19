from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path
import random
import statistics
import subprocess
import traceback
from typing import Any
import urllib.request

from opjax.pallas.laguna_dspark_profile import prometheus_values
from opjax.pallas.laguna_speculative import (
    DFLASH,
    DSPARK,
    PLAIN,
    VLLM_IMAGE,
    bind_trained_runtime_identity,
    canonical_sha256,
    ordered_replay_prompt_ids,
    replay_endpoint_headers,
    runtime_custom_all_reduce_disabled,
    runtime_gpu_memory_utilization,
    runtime_kv_cache_memory_bytes,
    runtime_tensor_parallel_size,
    runtime_tokenizer_path,
    run_replay_benchmark,
    validate_replay_attempt_receipt,
    validate_trained_replay_cells,
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
ARTIFACT_VOLUME = "opjax-laguna-speculative-artifacts-v1"
MODAL_ENVIRONMENT = "main"
MEASUREMENT_SOURCE_FILES = (
    Path(__file__),
    Path(__file__).with_name("laguna_docker_replay.py"),
)


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _measurement_sources() -> dict[str, str]:
    return {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in MEASUREMENT_SOURCE_FILES
    }


def _validate_attempt_context(
    receipt: dict[str, Any] | None,
    *,
    attempt_id: str,
    declared_gpu: str,
    expected_gpu_count: int,
    expected_tensor_parallel_size: int,
    expected_gpu_memory_utilization: float,
    expected_kv_cache_memory_bytes: int,
    expected_custom_all_reduce_disabled: bool,
    runtime: dict[str, Any] | None = None,
) -> None:
    if receipt is None:
        return
    payload = receipt["payload"]
    deployment = payload["deployment"]
    if (
        payload["attempt_id"] != attempt_id
        or payload["declared_gpu"] != declared_gpu
        or payload["gpu_count"] != expected_gpu_count
        or payload["tensor_parallel_size"] != expected_tensor_parallel_size
        or payload["gpu_memory_utilization"] != expected_gpu_memory_utilization
        or payload["kv_cache_memory_bytes"] != expected_kv_cache_memory_bytes
        or payload["disable_custom_all_reduce"]
        is not expected_custom_all_reduce_disabled
        or expected_tensor_parallel_size != expected_gpu_count
        or payload["image"] != VLLM_IMAGE
        or not set(_measurement_sources().values()).issubset(
            set(payload["measurement_sources"].values())
        )
    ):
        raise ValueError("LAGUNA_REPLAY_ATTEMPT_RECEIPT_CONTEXT_MISMATCH")
    if runtime is None:
        return
    if len((runtime.get("gpu") or {}).get("devices") or []) != expected_gpu_count:
        raise ValueError("LAGUNA_REPLAY_ATTEMPT_GPU_COUNT_MISMATCH")
    if runtime_tensor_parallel_size(runtime) != expected_tensor_parallel_size:
        raise ValueError("LAGUNA_REPLAY_ATTEMPT_TENSOR_PARALLEL_MISMATCH")
    if runtime_gpu_memory_utilization(runtime) != expected_gpu_memory_utilization:
        raise ValueError("LAGUNA_REPLAY_ATTEMPT_GPU_MEMORY_MISMATCH")
    if runtime_kv_cache_memory_bytes(runtime) != expected_kv_cache_memory_bytes:
        raise ValueError("LAGUNA_REPLAY_ATTEMPT_KV_CACHE_MEMORY_MISMATCH")
    if runtime_tokenizer_path(runtime) != payload["tokenizer"]["container_path"]:
        raise ValueError("LAGUNA_REPLAY_ATTEMPT_TOKENIZER_MISMATCH")
    if (
        runtime_custom_all_reduce_disabled(runtime)
        is not expected_custom_all_reduce_disabled
    ):
        raise ValueError("LAGUNA_REPLAY_ATTEMPT_ALL_REDUCE_MISMATCH")
    runtime_deployment = runtime.get("deployment") or {}
    expected_deployment = {
        "OPJAX_DEPLOYMENT_PROVIDER": deployment["provider"],
        "OPJAX_DEPLOYMENT_ID": deployment["id"],
        "OPJAX_DEPLOYMENT_ZONE": deployment["zone"],
        "OPJAX_DEPLOYMENT_INSTANCE_ID": deployment["instance_id"],
        "OPJAX_CONTAINER_INTERPRETER": payload["container_launcher"]["interpreter"],
    }
    if any(
        runtime_deployment.get(key) != value
        for key, value in expected_deployment.items()
    ):
        raise ValueError("LAGUNA_REPLAY_ATTEMPT_RUNTIME_MISMATCH")


def _validate_cell_artifacts(
    *,
    result: dict[str, Any],
    cell: str,
    attempt_id: str,
    output_root: Path,
    runtime_root: Path | None,
    required: bool,
) -> None:
    wrapper = result.get("cell_artifact_receipt")
    if not required and wrapper is None:
        return
    if not isinstance(wrapper, dict) or runtime_root is None:
        raise ValueError(f"LAGUNA_CELL_ARTIFACT_RECEIPT_MISSING:{cell}")
    payload = wrapper.get("payload") or {}
    expected = canonical_sha256(
        {key: value for key, value in payload.items() if key != "sha256"}
    )
    receipt_path = output_root / f"{cell}.artifact.json"
    raw = receipt_path.read_bytes()
    if (
        payload.get("sha256") != expected
        or wrapper.get("sha256") != expected
        or wrapper.get("file_sha256") != hashlib.sha256(raw).hexdigest()
        or json.loads(raw) != payload
        or payload.get("cell") != cell
        or payload.get("attempt_id") != attempt_id
        or payload.get("measurement_completed") is not True
        or payload.get("server_termination") != "stopped_after_measurement"
        or payload.get("pre_stop_returncode") is not None
        or payload.get("docker_stop_returncode") != 0
        or payload.get("server_process_returncode") not in {0, 137, 143}
        or payload.get("tokenizer")
        != (result.get("attempt_receipt") or {}).get("payload", {}).get("tokenizer")
    ):
        raise ValueError(f"LAGUNA_CELL_ARTIFACT_RECEIPT_INVALID:{cell}")
    roots = {
        "server_log": output_root,
        "gpu_csv": runtime_root,
        "runtime": runtime_root,
    }
    files = payload.get("files") or {}
    if set(files) != set(roots):
        raise ValueError(f"LAGUNA_CELL_ARTIFACT_FILE_SET_INVALID:{cell}")
    for name, root in roots.items():
        record = files[name]
        path = root / record["path"]
        content = path.read_bytes()
        if (
            len(content) != record.get("bytes")
            or hashlib.sha256(content).hexdigest() != record.get("sha256")
        ):
            raise ValueError(f"LAGUNA_CELL_ARTIFACT_FILE_INVALID:{cell}:{name}")


def _get(url: str, *, headers: dict[str, str]) -> str:
    request = urllib.request.Request(url, headers=headers)
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


def _runtime_path(cell: str, attempt_id: str) -> str:
    if cell == "plain":
        return f"plain/{attempt_id}-released-plain/runtime.json"
    arm, depth = cell.split("-", maxsplit=1)
    if cell == "dspark-adaptive":
        return f"dspark/{attempt_id}-trained-dspark-adaptive-15/runtime.json"
    return f"{arm}/{attempt_id}-trained-{arm}-fixed-{depth}/runtime.json"


def _runtime_evidence(
    cell: str, attempt_id: str, runtime_root: Path | None = None
) -> tuple[dict[str, Any], str]:
    relative = _runtime_path(cell, attempt_id)
    if runtime_root is None:
        result = subprocess.run(
            (
                "uv",
                "run",
                "modal",
                "volume",
                "get",
                "-e",
                MODAL_ENVIRONMENT,
                ARTIFACT_VOLUME,
                relative,
                "-",
            ),
            check=False,
            capture_output=True,
        )
        if result.returncode != 0:
            raise ValueError(
                f"LAGUNA_RUNTIME_EVIDENCE_READ_FAILED:{cell}:"
                f"{result.stderr.decode(errors='replace')}"
            )
        raw = result.stdout
    else:
        raw = (runtime_root / relative).read_bytes()
    if not raw:
        raise ValueError(f"LAGUNA_RUNTIME_EVIDENCE_EMPTY:{cell}")
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def _bind_runtime(
    *,
    cell: str,
    result: dict[str, Any],
    selections: dict[str, dict[str, Any]],
    attempt_id: str,
    expected_gpu_name: str,
    expected_gpu_count: int,
    expected_tensor_parallel_size: int,
    expected_gpu_memory_utilization: float,
    expected_kv_cache_memory_bytes: int,
    expected_custom_all_reduce_disabled: bool,
    runtime_root: Path | None,
) -> dict[str, Any]:
    runtime, runtime_file_sha256 = _runtime_evidence(
        cell, attempt_id, runtime_root
    )
    gpu = runtime.get("gpu") or {}
    devices = gpu.get("devices") or []
    if runtime.get("attempt_id") != attempt_id:
        raise ValueError(f"LAGUNA_RUNTIME_ATTEMPT_MISMATCH:{cell}")
    if runtime.get("declared_gpu") != expected_gpu_name:
        raise ValueError(f"LAGUNA_RUNTIME_DECLARED_GPU_MISMATCH:{cell}")
    if len(devices) != expected_gpu_count or not all(
        _gpu_name_matches(expected_gpu_name, str(device.get("name", "")))
        for device in devices
    ):
        raise ValueError(f"LAGUNA_RUNTIME_OBSERVED_GPU_MISMATCH:{cell}:{devices}")
    if runtime_tensor_parallel_size(runtime) != expected_tensor_parallel_size:
        raise ValueError(f"LAGUNA_RUNTIME_TENSOR_PARALLEL_MISMATCH:{cell}")
    if runtime_gpu_memory_utilization(runtime) != expected_gpu_memory_utilization:
        raise ValueError(f"LAGUNA_RUNTIME_GPU_MEMORY_MISMATCH:{cell}")
    if runtime_kv_cache_memory_bytes(runtime) != expected_kv_cache_memory_bytes:
        raise ValueError(f"LAGUNA_RUNTIME_KV_CACHE_MEMORY_MISMATCH:{cell}")
    if (
        runtime_custom_all_reduce_disabled(runtime)
        is not expected_custom_all_reduce_disabled
    ):
        raise ValueError(f"LAGUNA_RUNTIME_ALL_REDUCE_MISMATCH:{cell}")
    selection = selections.get(result["arm"])
    bound = bind_trained_runtime_identity(
        result=result,
        runtime=runtime,
        runtime_file_sha256=runtime_file_sha256,
        selection=selection,
    )
    bound["measurement_sources"] = _measurement_sources()
    bound["result_sha256"] = canonical_sha256(
        {key: value for key, value in bound.items() if key != "result_sha256"}
    )
    return bound


def _gpu_name_matches(declared: str, observed: str) -> bool:
    observed_normalized = observed.lower().replace(" ", "")
    return all(
        token.lower() in observed_normalized
        for token in declared.replace("_", "-").split("-")
        if token
    )


def _gpu_platform(gpu: dict[str, Any]) -> dict[str, Any]:
    devices = gpu.get("devices") or []
    if not devices:
        raise ValueError("LAGUNA_REPLAY_GPU_COUNT_INVALID:0")
    platform = {
        "count": len(devices),
        "devices": [
            {
                "name": device.get("name"),
                "driver_version": device.get("driver_version"),
                "memory_total_mib": device.get("memory_total_mib"),
                "compute_capability": device.get("compute_capability"),
            }
            for device in devices
        ],
    }
    if any(
        value in {None, ""}
        for device in platform["devices"]
        for value in device.values()
    ):
        raise ValueError(f"LAGUNA_REPLAY_GPU_PLATFORM_INVALID:{platform}")
    return platform


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
    ratio = _cluster_bootstrap_ratio(
        plain_rows=plain_rows,
        result_rows={row["prompt_id"]: row for row in result["records"]},
        prompt_ids=[row["prompt_id"] for row in matches],
    )
    return {
        "requests": len(result["records"]),
        "wall_s": result["wall_s"],
        "output_tps": result["output_tps"],
        "completion_tokens": result["completion_tokens"],
        "fixed_request_wall_speedup": plain["wall_s"] / result["wall_s"],
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
        "clustered_plain_over_cell_latency_on_matches": ratio,
        "speculation": {
            "drafted_tokens": drafted,
            "accepted_tokens": accepted,
            "acceptance_rate": accepted / drafted if drafted else None,
            "accepted_tokens_per_round": accepted / rounds if rounds else None,
        },
        "result_sha256": result["result_sha256"],
        "runtime_evidence": result["runtime_evidence"],
    }


def _cluster_bootstrap_ratio(
    *,
    plain_rows: dict[str, dict[str, Any]],
    result_rows: dict[str, dict[str, Any]],
    prompt_ids: list[str],
    samples: int = 10_000,
) -> dict[str, float | int] | None:
    grouped: dict[str, list[str]] = {}
    for prompt_id in prompt_ids:
        grouped.setdefault(plain_rows[prompt_id]["trajectory"], []).append(prompt_id)
    trajectories = sorted(grouped)
    if not trajectories:
        return None

    def ratio(selected: list[str]) -> float:
        ids = [prompt_id for name in selected for prompt_id in grouped[name]]
        return sum(plain_rows[prompt_id]["elapsed_s"] for prompt_id in ids) / sum(
            result_rows[prompt_id]["elapsed_s"] for prompt_id in ids
        )

    rng = random.Random(0)
    draws = sorted(
        ratio([rng.choice(trajectories) for _ in trajectories]) for _ in range(samples)
    )
    return {
        "ratio": ratio(trajectories),
        "ci95_low": draws[round((samples - 1) * 0.025)],
        "ci95_high": draws[round((samples - 1) * 0.975)],
        "trajectory_clusters": len(trajectories),
        "prompts": len(prompt_ids),
        "bootstrap_samples": samples,
    }


def _summary(
    *,
    split: str,
    corpus: dict[str, Any],
    selections: dict[str, dict[str, Any]],
    results: dict[str, dict[str, Any]],
    output_root: Path,
    attempt_id: str,
) -> dict[str, Any]:
    plain = results["plain"]
    hardware = {
        canonical_sha256(
            _gpu_platform(result.get("runtime_evidence", {}).get("gpu") or {})
        )
        for result in results.values()
    }
    if len(hardware) != 1:
        raise ValueError(f"LAGUNA_REPLAY_HARDWARE_MISMATCH:{sorted(hardware)}")
    attempts = {
        result.get("runtime_evidence", {}).get("attempt_id")
        for result in results.values()
    }
    if attempts != {attempt_id}:
        raise ValueError(f"LAGUNA_REPLAY_ATTEMPT_MISMATCH:{sorted(map(str, attempts))}")
    common_runtime_fields = (
        "image",
        "vllm_observed_build",
        "execution_sources",
        "deployment",
    )
    runtime_contracts = {
        canonical_sha256(
            {
                key: result["runtime_evidence"].get(key)
                for key in common_runtime_fields
            }
        )
        for result in results.values()
    }
    if len(runtime_contracts) != 1:
        raise ValueError("LAGUNA_REPLAY_RUNTIME_CONTRACT_MISMATCH")
    gpu_receipts = {
        canonical_sha256(result["runtime_evidence"]["gpu"])
        for result in results.values()
    }
    if len(gpu_receipts) != 1:
        raise ValueError("LAGUNA_REPLAY_GPU_DEVICE_MISMATCH")
    measurement_sources = {
        canonical_sha256(result.get("measurement_sources") or {})
        for result in results.values()
    }
    if measurement_sources != {canonical_sha256(_measurement_sources())}:
        raise ValueError("LAGUNA_REPLAY_MEASUREMENT_SOURCE_MISMATCH")
    tensor_parallel_sizes = {
        runtime_tensor_parallel_size(result["runtime_evidence"])
        for result in results.values()
    }
    if tensor_parallel_sizes != {len(plain["runtime_evidence"]["gpu"]["devices"])}:
        raise ValueError("LAGUNA_REPLAY_TENSOR_PARALLEL_MISMATCH")
    gpu_memory_utilizations = {
        runtime_gpu_memory_utilization(result["runtime_evidence"])
        for result in results.values()
    }
    attempt_gpu_memory = (plain.get("attempt_receipt") or {}).get("payload", {}).get(
        "gpu_memory_utilization"
    )
    if gpu_memory_utilizations != {attempt_gpu_memory}:
        raise ValueError("LAGUNA_REPLAY_GPU_MEMORY_MISMATCH")
    kv_cache_memory_values = {
        runtime_kv_cache_memory_bytes(result["runtime_evidence"])
        for result in results.values()
    }
    attempt_kv_cache_memory = (plain.get("attempt_receipt") or {}).get(
        "payload", {}
    ).get("kv_cache_memory_bytes")
    if kv_cache_memory_values != {attempt_kv_cache_memory}:
        raise ValueError("LAGUNA_REPLAY_KV_CACHE_MEMORY_MISMATCH")
    all_reduce_disabled = {
        runtime_custom_all_reduce_disabled(result["runtime_evidence"])
        for result in results.values()
    }
    attempt_all_reduce = (plain.get("attempt_receipt") or {}).get("payload", {}).get(
        "disable_custom_all_reduce"
    )
    if all_reduce_disabled != {attempt_all_reduce}:
        raise ValueError("LAGUNA_REPLAY_ALL_REDUCE_MISMATCH")
    attempt_receipts = {
        canonical_sha256(result.get("attempt_receipt") or {})
        for result in results.values()
    }
    if len(attempt_receipts) != 1:
        raise ValueError("LAGUNA_REPLAY_ATTEMPT_RECEIPT_MISMATCH")
    summary = {
        "schema_version": 1,
        "kind": "opjax_laguna_trained_replay_summary",
        "split": split,
        "corpus_sha256": corpus["release_sha256"],
        "selection_sha256": {
            arm: value["sha256"] for arm, value in selections.items()
        },
        "attempt_id": attempt_id,
        "attempt_receipt": plain.get("attempt_receipt"),
        "measurement_sources": _measurement_sources(),
        "runtime_contract": {
            key: plain["runtime_evidence"].get(key)
            for key in common_runtime_fields
        },
        "tensor_parallel_size": tensor_parallel_sizes.pop(),
        "gpu_memory_utilization": gpu_memory_utilizations.pop(),
        "kv_cache_memory_bytes": kv_cache_memory_values.pop(),
        "disable_custom_all_reduce": all_reduce_disabled.pop(),
        "gpu_platform": _gpu_platform(plain["runtime_evidence"]["gpu"]),
        "gpu_device_receipts": {
            cell: result["runtime_evidence"]["gpu"] for cell, result in results.items()
        },
        "cells": {
            cell: _cell_summary(result, plain) for cell, result in results.items()
        },
        "files": {
            cell: hashlib.sha256((output_root / f"{cell}.json").read_bytes()).hexdigest()
            for cell in results
        },
        "paired_models": _paired_models(results),
    }
    if split == "calibration":
        summary["depth_common_exact"] = {
            arm: _depth_common_exact(results, arm) for arm in (DFLASH, DSPARK)
        }
    summary["sha256"] = canonical_sha256(summary)
    return summary


def _depth_common_exact(
    results: dict[str, dict[str, Any]], arm: str
) -> dict[str, Any]:
    cells = sorted(
        (
            cell
            for cell in results
            if cell.startswith(f"{arm}-")
            and cell.removeprefix(f"{arm}-").isdigit()
        ),
        key=lambda cell: int(cell.removeprefix(f"{arm}-")),
    )
    expected = [f"{arm}-{depth}" for depth in (4, 8, 12, 15)]
    if cells != expected or "plain" not in results:
        raise ValueError(f"LAGUNA_DEPTH_CELL_SET_INVALID:{arm}:{cells}")
    plain_rows = {row["prompt_id"]: row for row in results["plain"]["records"]}
    cell_rows = {
        cell: {row["prompt_id"]: row for row in results[cell]["records"]}
        for cell in cells
    }
    prompt_ids = sorted(plain_rows)
    if any(set(rows) != set(prompt_ids) for rows in cell_rows.values()):
        raise ValueError(f"LAGUNA_DEPTH_PROMPT_SET_INVALID:{arm}")
    common = [
        prompt_id
        for prompt_id in prompt_ids
        if all(
            cell_rows[cell][prompt_id]["completion_token_ids"]
            == plain_rows[prompt_id]["completion_token_ids"]
            for cell in cells
        )
    ]
    if not common:
        raise ValueError(f"LAGUNA_DEPTH_COMMON_EXACT_EMPTY:{arm}")
    return {
        "policy": "compare only prompts whose token IDs equal plain at every fixed depth",
        "prompts": len(common),
        "trajectory_clusters": len(
            {plain_rows[prompt_id]["trajectory"] for prompt_id in common}
        ),
        "prompt_ids_sha256": canonical_sha256(common),
        "cells": {
            cell: {
                "plain_over_cell_latency": _cluster_bootstrap_ratio(
                    plain_rows=plain_rows,
                    result_rows=cell_rows[cell],
                    prompt_ids=common,
                ),
                "result_sha256": results[cell]["result_sha256"],
                "fixed_request_wall_s": results[cell]["wall_s"],
            }
            for cell in cells
        },
    }


def _paired_models(
    results: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    dflash_cells = [cell for cell in results if cell.startswith("dflash-")]
    dspark_cells = [cell for cell in results if cell.startswith("dspark-")]
    if len(dflash_cells) != 1 or len(dspark_cells) != 1 or "plain" not in results:
        return None
    dflash = results[dflash_cells[0]]
    dspark = results[dspark_cells[0]]
    plain_rows = {row["prompt_id"]: row for row in results["plain"]["records"]}
    dflash_rows = {row["prompt_id"]: row for row in dflash["records"]}
    dspark_rows = {row["prompt_id"]: row for row in dspark["records"]}
    common = [
        prompt_id
        for prompt_id, plain in plain_rows.items()
        if dflash_rows[prompt_id]["completion_token_ids"]
        == plain["completion_token_ids"]
        and dspark_rows[prompt_id]["completion_token_ids"]
        == plain["completion_token_ids"]
    ]
    return {
        "policy": "compare only prompts whose token IDs equal plain in both arms",
        "prompts": len(common),
        "dflash_cell": dflash_cells[0],
        "dspark_cell": dspark_cells[0],
        "dflash_over_dspark_latency": _cluster_bootstrap_ratio(
            plain_rows=dflash_rows,
            result_rows=dspark_rows,
            prompt_ids=common,
        ),
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
    parser.add_argument("--attempt-id", required=True)
    parser.add_argument("--expected-gpu-name", required=True)
    parser.add_argument("--expected-gpu-count", type=int, default=1)
    parser.add_argument("--expected-tensor-parallel-size", type=int, default=1)
    parser.add_argument("--expected-gpu-memory-utilization", type=float, default=0.82)
    parser.add_argument("--expected-kv-cache-memory-bytes", type=int, required=True)
    parser.add_argument("--expected-custom-all-reduce-disabled", action="store_true")
    parser.add_argument("--runtime-root", type=Path)
    parser.add_argument("--attempt-receipt", type=Path)
    parser.add_argument("--endpoint")
    parser.add_argument("--defer-summary", action="store_true")
    parser.add_argument("--finalize-existing", action="store_true")
    parser.add_argument("--max-workers", type=int, default=4)
    args = parser.parse_args()
    selected_cells = [item for item in args.cells.split(",") if item]
    validate_trained_replay_cells(
        selected_cells,
        known_cells=set(ENDPOINTS),
        endpoint=args.endpoint,
        defer_summary=args.defer_summary,
    )
    attempt_receipt = validate_replay_attempt_receipt(args.attempt_receipt)
    _validate_attempt_context(
        attempt_receipt,
        attempt_id=args.attempt_id,
        declared_gpu=args.expected_gpu_name,
        expected_gpu_count=args.expected_gpu_count,
        expected_tensor_parallel_size=args.expected_tensor_parallel_size,
        expected_gpu_memory_utilization=args.expected_gpu_memory_utilization,
        expected_kv_cache_memory_bytes=args.expected_kv_cache_memory_bytes,
        expected_custom_all_reduce_disabled=args.expected_custom_all_reduce_disabled,
    )
    manifest = json.loads(args.corpus_manifest.read_text())
    corpus = _filtered_corpus(
        json.loads(args.corpus.read_text()),
        split=args.split,
        task_ids=manifest["task_ids"][args.split],
    )
    selections = {arm: _selection(args.selection_root, arm) for arm in (DFLASH, DSPARK)}

    def load_complete(cell: str) -> dict[str, Any] | None:
        path = args.output_root / f"{cell}.json"
        if not path.is_file():
            return None
        result = json.loads(path.read_text())
        runtime, runtime_file_sha256 = _runtime_evidence(
            cell, args.attempt_id, args.runtime_root
        )
        _validate_attempt_context(
            attempt_receipt,
            attempt_id=args.attempt_id,
            declared_gpu=args.expected_gpu_name,
            expected_gpu_count=args.expected_gpu_count,
            expected_tensor_parallel_size=args.expected_tensor_parallel_size,
            expected_gpu_memory_utilization=args.expected_gpu_memory_utilization,
            expected_kv_cache_memory_bytes=args.expected_kv_cache_memory_bytes,
            expected_custom_all_reduce_disabled=(
                args.expected_custom_all_reduce_disabled
            ),
            runtime=runtime,
        )
        observed_devices = (runtime.get("gpu") or {}).get("devices") or []
        expected_runtime_sha256 = canonical_sha256(
            {key: value for key, value in runtime.items() if key != "sha256"}
        )
        expected_hash = canonical_sha256(
            {key: value for key, value in result.items() if key != "result_sha256"}
        )
        expected_arm = ENDPOINTS[cell][0]
        expected_identity = (
            {"plain_target": "poolside/Laguna-XS-2.1"}
            if expected_arm == PLAIN
            else selections[expected_arm]
        )
        if (
            result.get("result_sha256") != expected_hash
            or result.get("cell") != cell
            or result.get("arm") != expected_arm
            or result.get("model_identity") != expected_identity
            or result.get("corpus_sha256") != corpus["release_sha256"]
            or len(result.get("records", [])) != corpus["counts"]["prompts"]
            or ordered_replay_prompt_ids(result.get("records"))
            != ordered_replay_prompt_ids(corpus["records"])
            or not result.get("runtime_evidence", {}).get("runtime_sha256")
            or runtime.get("sha256") != expected_runtime_sha256
            or result.get("runtime_evidence", {}).get("runtime_sha256")
            != runtime.get("sha256")
            or result.get("runtime_evidence", {}).get("runtime_file_sha256")
            != runtime_file_sha256
            or result.get("runtime_evidence", {}).get("attempt_id") != args.attempt_id
            or runtime.get("attempt_id") != args.attempt_id
            or runtime.get("declared_gpu") != args.expected_gpu_name
            or len(observed_devices) != args.expected_gpu_count
            or not all(
                _gpu_name_matches(
                    args.expected_gpu_name, str(device.get("name", ""))
                )
                for device in observed_devices
            )
            or canonical_sha256(result.get("runtime_evidence", {}).get("gpu") or {})
            != canonical_sha256(runtime.get("gpu") or {})
            or result.get("measurement_sources") != _measurement_sources()
            or result.get("attempt_receipt") != attempt_receipt
        ):
            raise ValueError(f"LAGUNA_TRAINED_REPLAY_EXISTING_INVALID:{cell}")
        _validate_cell_artifacts(
            result=result,
            cell=cell,
            attempt_id=args.attempt_id,
            output_root=args.output_root,
            runtime_root=args.runtime_root,
            required=attempt_receipt is not None,
        )
        return result

    if args.finalize_existing:
        results = {}
        for cell in selected_cells:
            result = load_complete(cell)
            if result is None:
                raise ValueError(f"LAGUNA_TRAINED_REPLAY_EXISTING_MISSING:{cell}")
            results[cell] = result
        summary = _summary(
            split=args.split,
            corpus=corpus,
            selections=selections,
            results=results,
            output_root=args.output_root,
            attempt_id=args.attempt_id,
        )
        _write(args.output_root / "summary.json", summary)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return

    def execute(cell: str) -> tuple[str, dict[str, Any]]:
        existing = load_complete(cell)
        if existing is not None:
            return cell, existing
        arm, default_endpoint = ENDPOINTS[cell]
        endpoint = args.endpoint or default_endpoint
        headers = replay_endpoint_headers(endpoint, modal_proxy_headers)
        identity = (
            {"plain_target": "poolside/Laguna-XS-2.1"}
            if arm == PLAIN
            else selections[arm]
        )
        _get(f"{endpoint}/health", headers=headers)
        warm_replay_endpoint(
            base_url=endpoint,
            headers=headers,
            corpus=corpus,
            max_tokens=8192,
        )
        before = _get(f"{endpoint}/metrics", headers=headers)
        result = run_replay_benchmark(
            arm=arm,
            base_url=endpoint,
            headers=headers,
            corpus=corpus,
            concurrency=1,
            max_tokens=8192,
            warmup=False,
            model_identity=identity,
        )
        after = _get(f"{endpoint}/metrics", headers=headers)
        result["cell"] = cell
        result["endpoint"] = endpoint
        result["prometheus_delta"] = _delta(before, after)
        result = _bind_runtime(
            cell=cell,
            result=result,
            selections=selections,
            attempt_id=args.attempt_id,
            expected_gpu_name=args.expected_gpu_name,
            expected_gpu_count=args.expected_gpu_count,
            expected_tensor_parallel_size=args.expected_tensor_parallel_size,
            expected_gpu_memory_utilization=args.expected_gpu_memory_utilization,
            expected_kv_cache_memory_bytes=args.expected_kv_cache_memory_bytes,
            expected_custom_all_reduce_disabled=(
                args.expected_custom_all_reduce_disabled
            ),
            runtime_root=args.runtime_root,
        )
        result["attempt_receipt"] = attempt_receipt
        result["result_sha256"] = canonical_sha256(
            {key: value for key, value in result.items() if key != "result_sha256"}
        )
        _write(args.output_root / f"{cell}.json", result)
        return cell, result

    results = {}
    failures = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {executor.submit(execute, cell): cell for cell in selected_cells}
        for future in concurrent.futures.as_completed(futures):
            cell = futures[future]
            try:
                completed_cell, result = future.result()
                results[completed_cell] = result
                print(f"LAGUNA_REPLAY_CELL_COMPLETE:{completed_cell}", flush=True)
            except Exception as error:
                failures[cell] = {
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "traceback": traceback.format_exc(),
                }
                print(f"LAGUNA_REPLAY_CELL_FAILED:{cell}:{error}", flush=True)
    if failures:
        failure_payload = {
            "schema_version": 1,
            "kind": "opjax_laguna_trained_replay_failures",
            "split": args.split,
            "corpus_sha256": corpus["release_sha256"],
            "attempt_id": args.attempt_id,
            "failures": failures,
        }
        failure_payload["sha256"] = canonical_sha256(failure_payload)
        _write(args.output_root / "failures.json", failure_payload)
        raise RuntimeError(f"LAGUNA_TRAINED_REPLAY_INCOMPLETE:{sorted(failures)}")
    if args.defer_summary:
        return
    failure_path = args.output_root / "failures.json"
    if failure_path.is_file():
        archive_root = args.output_root / "failure-attempts"
        archive_root.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256(failure_path.read_bytes()).hexdigest()
        failure_path.replace(archive_root / f"{digest}.json")
    summary = _summary(
        split=args.split,
        corpus=corpus,
        selections=selections,
        results=results,
        output_root=args.output_root,
        attempt_id=args.attempt_id,
    )
    _write(args.output_root / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
