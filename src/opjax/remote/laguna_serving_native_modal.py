from __future__ import annotations

import hashlib
import gzip
import json
import os
from pathlib import Path
import signal
import subprocess
import time
from typing import Any

import modal
import numpy as np

from opjax.pallas.laguna_serving_native import (
    captured_prompt_token_ids,
    reconstruct_committed_sample,
    reconstruct_fixed_sample,
    select_final_requests,
    serving_prefix_ends,
    validate_sample,
    write_sample,
)
from opjax.pallas.laguna_speculative import (
    BASH_TOOL,
    DFLASH,
    TARGET_ID,
    VLLM_IMAGE,
    canonical_sha256,
    server_command,
)
from opjax.remote.config import (
    HF_CACHE_DIR,
    HF_CACHE_VOLUME_NAME,
    MODAL_ENVIRONMENT,
    MODAL_SECRET_NAME,
    MODAL_VOLUME_VERSION,
    REMOTE_ENV,
)
from opjax.remote.laguna_vllm_conformance import _request, _wait_ready


APP_NAME = "opjax-laguna-serving-native-v1"
ROOT = Path("/mnt/serving-native")
TRAINING_ROOT = Path("/mnt/training")
PORT = 8000

app = modal.App(APP_NAME)
cache = modal.Volume.from_name(
    HF_CACHE_VOLUME_NAME,
    environment_name=MODAL_ENVIRONMENT,
    create_if_missing=True,
    version=MODAL_VOLUME_VERSION,
)
artifacts = modal.Volume.from_name(
    "opjax-laguna-serving-native-v1",
    environment_name=MODAL_ENVIRONMENT,
    create_if_missing=True,
    version=MODAL_VOLUME_VERSION,
)
training = modal.Volume.from_name(
    "opjax-laguna-speculator-training-v1",
    environment_name=MODAL_ENVIRONMENT,
    create_if_missing=True,
    version=MODAL_VOLUME_VERSION,
)
secret = modal.Secret.from_name(MODAL_SECRET_NAME, environment_name=MODAL_ENVIRONMENT)

image = (
    modal.Image.from_registry(VLLM_IMAGE, add_python="3.12")
    .entrypoint([])
    .uv_pip_install("huggingface-hub==1.4.1", "numpy==2.4.4")
    .env(
        {
            **REMOTE_ENV,
            "OPJAX_SPEC_ARTIFACT_ROOT": str(ROOT / "runtime"),
            "OPJAX_SPEC_ARTIFACT_VOLUME": "opjax-laguna-serving-native-v1",
            "OPJAX_SPEC_MODAL_ENVIRONMENT": MODAL_ENVIRONMENT,
        }
    )
    .add_local_python_source("opjax")
    .add_local_file(
        "data/pallas/runs/laguna-speculative-v1/replay-corpus.json",
        "/opt/opjax/replay-corpus.json",
    )
    .add_local_file(
        "data/pallas/corpora/laguna-speculator-v1/manifest.json",
        "/opt/opjax/split-manifest.json",
    )
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_artifact_manifest(root: Path, *, runtime: dict[str, Any]) -> None:
    files = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        if path.name == "artifact-manifest.json":
            continue
        files.append(
            {
                "path": str(path.relative_to(root)),
                "sha256": _sha256(path),
                "bytes": path.stat().st_size,
            }
        )
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "files": files,
        "runtime": runtime,
    }
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    (root / "artifact-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _validate_artifact_manifest(path: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if canonical_sha256(
        {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    ) != manifest.get("manifest_sha256"):
        raise RuntimeError(f"SERVING_NATIVE_ARTIFACT_MANIFEST_HASH:{path}")
    root = path.parent
    seen = set()
    for record in manifest.get("files", []):
        relative = record.get("path")
        candidate = (root / str(relative)).resolve()
        if (
            not isinstance(relative, str)
            or relative in seen
            or not candidate.is_relative_to(root.resolve())
            or not candidate.is_file()
            or candidate.stat().st_size != record.get("bytes")
            or _sha256(candidate) != record.get("sha256")
        ):
            raise RuntimeError(f"SERVING_NATIVE_ARTIFACT_FILE_INVALID:{path}:{relative}")
        seen.add(relative)
    actual = {
        str(candidate.relative_to(root))
        for candidate in root.rglob("*")
        if candidate.is_file() and candidate != path
    }
    if actual != seen:
        raise RuntimeError(f"SERVING_NATIVE_ARTIFACT_FILE_SET:{path}")
    return manifest


def _replace_argument(command: list[str], name: str, value: str) -> None:
    index = command.index(name) + 1
    command[index] = value


def _profile_receipt(root: Path) -> dict[str, Any]:
    traces = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and (path.name.endswith(".json") or path.name.endswith(".json.gz"))
    )
    if not traces:
        raise RuntimeError(f"SERVING_NATIVE_PROFILE_MISSING:{root}")
    records = []
    total_launches = 0
    for path in traces:
        launches = 0
        handle_context = (
            gzip.open(path, "rb") if path.name.endswith(".gz") else path.open("rb")
        )
        with handle_context as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                launches += chunk.count(b'"cudaLaunchKernel"')
                launches += chunk.count(b'"cuLaunchKernelEx"')
        records.append(
            {
                "path": str(path.relative_to(root)),
                "sha256": _sha256(path),
                "bytes": path.stat().st_size,
                "cuda_launch_events": launches,
            }
        )
        total_launches += launches
    if total_launches < 1:
        raise RuntimeError(f"SERVING_NATIVE_PROFILE_CUDA_MISSING:{root}")
    return {"traces": records, "cuda_launch_events": total_launches}


def _preserve_or_archive_profile(
    *, run_root: Path, profile_root: Path
) -> dict[str, Any] | None:
    if not any(profile_root.iterdir()):
        return None
    try:
        return _profile_receipt(profile_root)
    except RuntimeError:
        archive = run_root / "attempts" / f"profiles-{time.time_ns()}"
        archive.parent.mkdir(parents=True, exist_ok=True)
        os.replace(profile_root, archive)
        profile_root.mkdir()
        return None


def _request_payload(record: dict[str, Any], max_tokens: int) -> dict[str, Any]:
    return {
        "model": TARGET_ID,
        "messages": record["messages"],
        "tools": [BASH_TOOL],
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "seed": 0,
        "chat_template_kwargs": {"enable_thinking": True},
        "return_token_ids": True,
    }


def _load_tokenized_sample(
    tokenized_root: Path, split: str, sample_id: str
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    sample_root = tokenized_root / split / sample_id
    manifest_path = sample_root / "manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError(f"SERVING_NATIVE_TOKEN_MANIFEST_MISSING:{sample_id}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if canonical_sha256(
        {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    ) != manifest.get("manifest_sha256"):
        raise RuntimeError(f"SERVING_NATIVE_TOKEN_MANIFEST_HASH:{sample_id}")
    arrays = {}
    for name in ("input_ids", "attention_mask", "loss_mask"):
        record = manifest["files"][name]
        path = sample_root / record["path"]
        if not path.is_file() or _sha256(path) != record["sha256"]:
            raise RuntimeError(f"SERVING_NATIVE_TOKEN_FILE_HASH:{sample_id}:{name}")
        value = np.load(path, allow_pickle=False)
        if list(value.shape) != record["shape"] or str(value.dtype) != record["dtype"]:
            raise RuntimeError(f"SERVING_NATIVE_TOKEN_FILE_CONTRACT:{sample_id}:{name}")
        arrays[name] = value
    if not np.array_equal(arrays["attention_mask"], np.ones_like(arrays["attention_mask"])):
        raise RuntimeError(f"SERVING_NATIVE_TOKEN_ATTENTION_MASK:{sample_id}")
    return arrays["input_ids"], arrays["loss_mask"], manifest


def _fixed_request_payload(
    input_ids: np.ndarray, end: int, *, cache_salt: str
) -> dict[str, Any]:
    return {
        "model": TARGET_ID,
        "prompt": input_ids[:end].astype(np.int64).tolist(),
        "temperature": 0.0,
        "max_tokens": 1,
        "seed": 0,
        "return_token_ids": True,
        "cache_salt": cache_salt,
    }


def _target_round_receipts(session_root: Path, *, after_round: int) -> list[dict[str, int]]:
    ledger_path = session_root / "target-ledger.jsonl"
    if not ledger_path.is_file():
        return []
    records = [json.loads(line) for line in ledger_path.read_text(encoding="utf-8").splitlines()]
    positions_by_round: dict[int, Path] = {}
    for record in records:
        round_id = record.get("round")
        if (
            isinstance(round_id, int)
            and round_id > after_round
            and record.get("name") == "target_positions"
        ):
            if round_id in positions_by_round:
                raise RuntimeError(f"SERVING_NATIVE_TARGET_ROUND_DUPLICATE:{round_id}")
            path = session_root / str(record.get("path"))
            if not path.is_file() or _sha256(path) != record.get("sha256"):
                raise RuntimeError(f"SERVING_NATIVE_TARGET_ROUND_HASH:{round_id}")
            positions_by_round[round_id] = path
    result = []
    for round_id, path in sorted(positions_by_round.items()):
        positions = np.load(path, allow_pickle=False).reshape(-1)
        if positions.size < 1:
            raise RuntimeError(f"SERVING_NATIVE_TARGET_ROUND_EMPTY:{round_id}")
        result.append(
            {
                "round": round_id,
                "position_min": int(positions.min()),
                "position_max": int(positions.max()),
                "rows": int(positions.size),
            }
        )
    return result


def _validate_prefix_cache_start(
    *, prefix_index: int, processed_start: int, previous_end: int
) -> None:
    expected = (
        0
        if prefix_index == 0
        else max(0, (previous_end // 16) * 16 - 16)
    )
    if processed_start != expected:
        raise RuntimeError(
            f"SERVING_NATIVE_PREFIX_CACHE_START:{prefix_index}:"
            f"{processed_start}:{expected}:{previous_end}"
        )


def _capture_fixed_records(
    *, split: str, run_id: str, shard_index: int, shard_count: int
) -> dict[str, Any]:
    tokenized_root = ROOT / run_id / "tokenized"
    token_release_path = tokenized_root / "release.json"
    token_release = json.loads(token_release_path.read_text(encoding="utf-8"))
    if canonical_sha256(
        {key: value for key, value in token_release.items() if key != "release_sha256"}
    ) != token_release.get("release_sha256"):
        raise RuntimeError("SERVING_NATIVE_TOKEN_RELEASE_HASH_INVALID")
    selected = token_release["splits"][split]["records"][shard_index::shard_count]
    run_root = ROOT / run_id / "fixed-shards" / f"{split}-{shard_index:03d}-of-{shard_count:03d}"
    raw_root = run_root / "raw" / split
    sample_root = run_root / "samples" / split
    profile_root = run_root / "profiles" / split
    raw_root.mkdir(parents=True, exist_ok=True)
    sample_root.mkdir(parents=True, exist_ok=True)
    profile_root.mkdir(parents=True, exist_ok=True)
    summary_path = run_root / f"capture-{split}.json"
    artifact_path = run_root / "artifact-manifest.json"
    if summary_path.exists() or artifact_path.exists():
        if not summary_path.is_file() or not artifact_path.is_file():
            raise RuntimeError(f"SERVING_NATIVE_COMPLETED_SHARD_PARTIAL:{run_root}")
        _validate_artifact_manifest(artifact_path)
        completed = json.loads(summary_path.read_text(encoding="utf-8"))
        if canonical_sha256(
            {key: value for key, value in completed.items() if key != "summary_sha256"}
        ) != completed.get("summary_sha256"):
            raise RuntimeError(f"SERVING_NATIVE_COMPLETED_SHARD_INVALID:{run_root}")
        return completed
    command = server_command(DFLASH, port=PORT, proposal_tokens=15, capture_target=True)
    _replace_argument(command, "--max-num-seqs", "1")
    command.extend(
        [
            "--enable-prefix-caching",
            "--block-size",
            "16",
            "--safetensors-load-strategy",
            "prefetch",
            "--enforce-eager",
            "--profiler-config",
            json.dumps(
                {
                    "profiler": "torch",
                    "torch_profiler_dir": str(profile_root),
                    "torch_profiler_with_stack": False,
                    "torch_profiler_record_shapes": True,
                    "torch_profiler_with_memory": True,
                    "torch_profiler_use_gzip": True,
                },
                sort_keys=True,
                separators=(",", ":"),
            ),
        ]
    )
    runtime_id = f"{run_id}-fixed-{split}-{shard_index:03d}-of-{shard_count:03d}"
    environment = {
        **os.environ,
        "OPJAX_LAGUNA_TARGET_CAPTURE_ROOT": str(raw_root),
        "OPJAX_SPEC_RUN_ID": runtime_id,
    }
    log_path = run_root / f"server-{split}.log"
    preserved_profile = _preserve_or_archive_profile(
        run_root=run_root, profile_root=profile_root
    )
    if log_path.exists():
        archive = run_root / "attempts" / f"server-{time.time_ns()}.log"
        archive.parent.mkdir(parents=True, exist_ok=True)
        os.replace(log_path, archive)
    runtime_path = ROOT / "runtime" / DFLASH / runtime_id / "runtime.json"
    if runtime_path.parent.exists():
        archive = run_root / "attempts" / f"runtime-{time.time_ns()}"
        archive.parent.mkdir(parents=True, exist_ok=True)
        os.replace(runtime_path.parent, archive)
    results = []
    profiled_prompt_id = selected[0]["id"] if preserved_profile else None
    started = time.time()
    with log_path.open("wb") as server_log:
        process = subprocess.Popen(
            command,
            env=environment,
            stdout=server_log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            _wait_ready(PORT, process, log_path=log_path)
            for record in selected:
                session = record["id"]
                session_root = raw_root / session
                sample_path = sample_root / session
                if sample_path.is_dir():
                    try:
                        results.append(validate_sample(sample_path))
                        continue
                    except Exception as error:
                        archive = run_root / "attempts" / f"sample-{session}-{time.time_ns()}"
                        archive.parent.mkdir(parents=True, exist_ok=True)
                        os.replace(sample_path, archive)
                        (archive / "archive.json").write_text(
                            json.dumps({"reason": str(error)}, sort_keys=True) + "\n",
                            encoding="utf-8",
                        )
                if session_root.exists():
                    archive = run_root / "attempts" / f"raw-{session}-{time.time_ns()}"
                    archive.parent.mkdir(parents=True, exist_ok=True)
                    os.replace(session_root, archive)
                input_ids, loss_mask, token_manifest = _load_tokenized_sample(
                    tokenized_root, split, session
                )
                prefix_ends = serving_prefix_ends(loss_mask, proposal_length=15)
                cache_salt = hashlib.sha256(
                    f"{run_id}:{split}:{session}".encode()
                ).hexdigest()
                (raw_root / "active.json").write_text(
                    json.dumps({"session": session}), encoding="utf-8"
                )
                prefix_receipts = []
                should_profile = profiled_prompt_id is None
                if should_profile:
                    profiled_prompt_id = session
                    _request(
                        f"http://127.0.0.1:{PORT}/start_profile",
                        payload={},
                        timeout=300,
                    )
                for prefix_index, end in enumerate(prefix_ends):
                    prior_round = (
                        prefix_receipts[-1]["target_rounds"][-1]["round"]
                        if prefix_receipts
                        else -1
                    )
                    response = json.loads(
                        _request(
                            f"http://127.0.0.1:{PORT}/v1/completions",
                            payload=_fixed_request_payload(
                                input_ids, end, cache_salt=cache_salt
                            ),
                            timeout=600,
                        )
                    )
                    target_rounds = _target_round_receipts(
                        session_root, after_round=prior_round
                    )
                    if not target_rounds:
                        raise RuntimeError(
                            f"SERVING_NATIVE_TARGET_ROUND_MISSING:{session}:{prefix_index}"
                        )
                    processed_start = target_rounds[0]["position_min"]
                    previous_end = 0 if prefix_index == 0 else prefix_ends[prefix_index - 1]
                    _validate_prefix_cache_start(
                        prefix_index=prefix_index,
                        processed_start=processed_start,
                        previous_end=previous_end,
                    )
                    prefix_receipts.append(
                        {
                            "index": prefix_index,
                            "end": end,
                            "previous_prefix_end": previous_end,
                            "processed_start": processed_start,
                            "cached_prefix_tokens": processed_start,
                            "target_rounds": target_rounds,
                            "response_sha256": canonical_sha256(response),
                        }
                    )
                    if should_profile and prefix_index == min(3, len(prefix_ends) - 1):
                        _request(
                            f"http://127.0.0.1:{PORT}/stop_profile",
                            payload={},
                            timeout=600,
                        )
                        should_profile = False
                (raw_root / "active.json").unlink()
                sample, metadata = reconstruct_fixed_sample(
                    session_root=session_root,
                    input_ids=input_ids,
                    loss_mask=loss_mask,
                )
                metadata.update(
                    {
                        "id": session,
                        "trajectory": record["trajectory"],
                        "split": split,
                        "token_manifest_sha256": token_manifest["manifest_sha256"],
                        "cache_salt_sha256": hashlib.sha256(cache_salt.encode()).hexdigest(),
                        "prefix_ends": prefix_ends,
                        "prefix_receipts": prefix_receipts,
                    }
                )
                metadata["metadata_sha256"] = canonical_sha256(
                    {key: value for key, value in metadata.items() if key != "metadata_sha256"}
                )
                temporary_sample = sample_root / f".{session}.{time.time_ns()}.tmp"
                manifest = write_sample(temporary_sample, sample=sample, metadata=metadata)
                result = {
                    "prompt_id": session,
                    "trajectory": record["trajectory"],
                    "manifest_sha256": manifest["manifest_sha256"],
                    "manifest_file_sha256": _sha256(
                        temporary_sample / "manifest.json"
                    ),
                    "token_manifest_sha256": token_manifest["manifest_sha256"],
                    "tokens": len(input_ids),
                }
                result["result_sha256"] = canonical_sha256(result)
                (temporary_sample / "capture-result.json").write_text(
                    json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
                )
                validate_sample(temporary_sample)
                os.replace(temporary_sample, sample_path)
                results.append(result)
                artifacts.commit()
        finally:
            (raw_root / "active.json").unlink(missing_ok=True)
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=30)
    if not runtime_path.is_file():
        raise RuntimeError(f"SERVING_NATIVE_RUNTIME_MISSING:{runtime_path}")
    profile = preserved_profile or _profile_receipt(profile_root)
    summary: dict[str, Any] = {
        "schema_version": 1,
        "kind": "opjax_laguna_serving_native_fixed_capture",
        "run_id": run_id,
        "split": split,
        "shard_index": shard_index,
        "shard_count": shard_count,
        "records": results,
        "record_count": len(results),
        "token_release_sha256": token_release["release_sha256"],
        "runtime": {
            "run_id": runtime_id,
            "path": str(runtime_path.relative_to(ROOT)),
            "sha256": _sha256(runtime_path),
        },
        "server_log": {"path": log_path.name, "sha256": _sha256(log_path)},
        "profiled_prompt_id": profiled_prompt_id,
        "profile_preserved_from_incomplete_attempt": preserved_profile is not None,
        "profile": profile,
        "wall_seconds": time.time() - started,
    }
    summary["summary_sha256"] = canonical_sha256(summary)
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_artifact_manifest(
        run_root,
        runtime={
            "command": command,
            "runtime_id": runtime_id,
            "token_release_sha256": token_release["release_sha256"],
            "token_release_file_sha256": _sha256(token_release_path),
            "capture_driver_sha256": _sha256(Path(__file__)),
            "reconstruction_source_sha256": _sha256(
                Path(write_sample.__code__.co_filename)
            ),
        },
    )
    artifacts.commit()
    return summary


def _rebuild_fixed_records(
    *, split: str, run_id: str, shard_index: int, shard_count: int
) -> dict[str, Any]:
    run_root = (
        ROOT
        / run_id
        / "fixed-shards"
        / f"{split}-{shard_index:03d}-of-{shard_count:03d}"
    )
    capture_summary_path = run_root / f"capture-{split}.json"
    capture_artifact_path = run_root / "artifact-manifest.json"
    old_artifact = _validate_artifact_manifest(capture_artifact_path)
    old_summary = json.loads(capture_summary_path.read_text(encoding="utf-8"))
    if canonical_sha256(
        {key: value for key, value in old_summary.items() if key != "summary_sha256"}
    ) != old_summary.get("summary_sha256"):
        raise RuntimeError(f"SERVING_NATIVE_REBUILD_SUMMARY_HASH:{run_root}")
    policy_name = "first-causal-observation-wins"
    rebuild_driver_sha256 = _sha256(Path(__file__))
    reconstruction_source_sha256 = _sha256(Path(write_sample.__code__.co_filename))
    policy_base = (
        ROOT
        / run_id
        / "rebuilt-fixed-shards"
        / f"{split}-{shard_index:03d}-of-{shard_count:03d}"
    )
    policy_root = policy_base / "policies" / policy_name
    policy_summary_path = policy_root / "policy-summary.json"
    policy_artifact_path = policy_root / "artifact-manifest.json"
    if policy_summary_path.is_file() and policy_artifact_path.is_file():
        completed_artifact = _validate_artifact_manifest(policy_artifact_path)
        completed = json.loads(policy_summary_path.read_text(encoding="utf-8"))
        if canonical_sha256(
            {key: value for key, value in completed.items() if key != "summary_sha256"}
        ) != completed.get("summary_sha256"):
            raise RuntimeError(f"SERVING_NATIVE_REBUILD_POLICY_HASH:{policy_root}")
        if (
            completed.get("rebuild_driver_sha256") != rebuild_driver_sha256
            or completed_artifact.get("runtime", {}).get("rebuild_driver_sha256")
            != rebuild_driver_sha256
            or completed.get("reconstruction_source_sha256")
            != reconstruction_source_sha256
            or completed_artifact.get("runtime", {}).get(
                "reconstruction_source_sha256"
            )
            != reconstruction_source_sha256
            or completed.get("source_capture_summary_sha256")
            != old_summary["summary_sha256"]
            or completed_artifact.get("runtime", {}).get(
                "source_capture_summary_sha256"
            )
            != old_summary["summary_sha256"]
            or completed.get("source_capture_artifact_manifest_sha256")
            != old_artifact["manifest_sha256"]
            or completed_artifact.get("runtime", {}).get(
                "source_capture_artifact_manifest_sha256"
            )
            != old_artifact["manifest_sha256"]
            or completed.get("superseded", {}).get("artifact_manifest_sha256")
            != old_artifact["manifest_sha256"]
        ):
            raise RuntimeError(f"SERVING_NATIVE_REBUILD_POLICY_DRIFT:{policy_root}")
        return completed
    tokenized_root = ROOT / run_id / "tokenized"
    token_release_path = tokenized_root / "release.json"
    token_release = json.loads(token_release_path.read_text(encoding="utf-8"))
    if canonical_sha256(
        {key: value for key, value in token_release.items() if key != "release_sha256"}
    ) != token_release.get("release_sha256"):
        raise RuntimeError("SERVING_NATIVE_REBUILD_TOKEN_RELEASE_HASH")
    selected = token_release["splits"][split]["records"][shard_index::shard_count]
    staging = policy_base / "policy-staging" / policy_name
    staging_samples = staging / "samples" / split
    staging_samples.mkdir(parents=True, exist_ok=True)
    stale_attempts = sorted(staging_samples.glob(".*.tmp"))
    if stale_attempts:
        attempt_root = policy_base / "policy-staging-attempts" / str(time.time_ns())
        attempt_root.mkdir(parents=True, exist_ok=False)
        for stale in stale_attempts:
            os.replace(stale, attempt_root / stale.name)
    records = []
    totals = {
        "overlap_rows": 0,
        "divergent_overlap_rows": 0,
        "divergent_loss_overlap_rows": 0,
    }
    for token_record in selected:
        session = token_record["id"]
        sample_path = run_root / "samples" / split / session
        old_result = validate_sample(sample_path)
        old_manifest = json.loads(
            (sample_path / "manifest.json").read_text(encoding="utf-8")
        )
        metadata = old_manifest["metadata"]
        receipts = metadata.get("prefix_receipts")
        prefix_ends = metadata.get("prefix_ends")
        if (
            not isinstance(receipts, list)
            or not isinstance(prefix_ends, list)
            or len(receipts) != len(prefix_ends)
        ):
            raise RuntimeError(f"SERVING_NATIVE_REBUILD_RECEIPTS:{session}")
        for index, receipt in enumerate(receipts):
            _validate_prefix_cache_start(
                prefix_index=index,
                processed_start=receipt["processed_start"],
                previous_end=receipt["previous_prefix_end"],
            )
            if receipt.get("end") != prefix_ends[index]:
                raise RuntimeError(f"SERVING_NATIVE_REBUILD_PREFIX_END:{session}:{index}")
        destination = staging_samples / session
        if destination.is_dir():
            result = validate_sample(destination)
            rebuilt_metadata = json.loads(
                (destination / "manifest.json").read_text(encoding="utf-8")
            )["metadata"]
            staged_matches = (
                rebuilt_metadata.get("feature_selection_policy")
                == "first_causal_observation_wins"
                and rebuilt_metadata.get("superseded_manifest_sha256")
                == old_result["manifest_sha256"]
                and rebuilt_metadata.get(
                    "source_capture_artifact_manifest_sha256"
                )
                == old_artifact["manifest_sha256"]
                and rebuilt_metadata.get("rebuild_driver_sha256")
                == rebuild_driver_sha256
                and rebuilt_metadata.get("reconstruction_source_sha256")
                == reconstruction_source_sha256
                and result.get("token_manifest_sha256")
                == token_record["manifest_sha256"]
            )
            if staged_matches:
                for key in totals:
                    totals[key] += int(rebuilt_metadata[key])
                records.append(result)
                continue
            stale_root = (
                policy_base
                / "policy-staging-attempts"
                / f"stale-{session}-{time.time_ns()}"
            )
            stale_root.parent.mkdir(parents=True, exist_ok=True)
            os.replace(destination, stale_root)
        input_ids, loss_mask, token_manifest = _load_tokenized_sample(
            tokenized_root, split, session
        )
        sample, rebuilt = reconstruct_fixed_sample(
            session_root=run_root / "raw" / split / session,
            input_ids=input_ids,
            loss_mask=loss_mask,
        )
        rebuilt.update(
            {
                "id": session,
                "trajectory": token_record["trajectory"],
                "split": split,
                "token_manifest_sha256": token_manifest["manifest_sha256"],
                "prefix_ends": prefix_ends,
                "prefix_receipts": receipts,
                "cache_salt_sha256": metadata["cache_salt_sha256"],
                "superseded_manifest_sha256": old_result["manifest_sha256"],
                "source_capture_artifact_manifest_sha256": old_artifact[
                    "manifest_sha256"
                ],
                "rebuild_driver_sha256": rebuild_driver_sha256,
                "reconstruction_source_sha256": reconstruction_source_sha256,
            }
        )
        rebuilt["metadata_sha256"] = canonical_sha256(
            {key: value for key, value in rebuilt.items() if key != "metadata_sha256"}
        )
        for key in totals:
            totals[key] += int(rebuilt[key])
        temporary = staging_samples / f".{session}.{time.time_ns()}.tmp"
        manifest = write_sample(temporary, sample=sample, metadata=rebuilt)
        result: dict[str, Any] = {
            "prompt_id": session,
            "trajectory": token_record["trajectory"],
            "manifest_sha256": manifest["manifest_sha256"],
            "manifest_file_sha256": _sha256(temporary / "manifest.json"),
            "token_manifest_sha256": token_manifest["manifest_sha256"],
            "tokens": len(input_ids),
        }
        result["result_sha256"] = canonical_sha256(result)
        (temporary / "capture-result.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        validate_sample(temporary)
        os.replace(temporary, destination)
        records.append(result)
        artifacts.commit()
    policy_summary = {
        key: value
        for key, value in old_summary.items()
        if key not in {"records", "record_count", "summary_sha256"}
    }
    policy_summary.update(
        {
            "records": records,
            "record_count": len(records),
            "feature_selection_policy": "first_causal_observation_wins",
            "overlap_divergence": totals,
            "rebuild_driver_sha256": rebuild_driver_sha256,
            "reconstruction_source_sha256": reconstruction_source_sha256,
            "source_capture_summary_sha256": old_summary["summary_sha256"],
            "source_capture_artifact_manifest_sha256": old_artifact[
                "manifest_sha256"
            ],
            "superseded": {
                "summary_file_sha256": _sha256(capture_summary_path),
                "summary_sha256": old_summary["summary_sha256"],
                "artifact_manifest_file_sha256": _sha256(capture_artifact_path),
                "artifact_manifest_sha256": old_artifact["manifest_sha256"],
            },
        }
    )
    policy_summary["summary_sha256"] = canonical_sha256(policy_summary)
    (staging / "policy-summary.json").write_text(
        json.dumps(policy_summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    runtime = dict(old_artifact["runtime"])
    runtime.update(
        {
            "feature_selection_policy": "first_causal_observation_wins",
            "rebuild_driver_sha256": rebuild_driver_sha256,
            "reconstruction_source_sha256": policy_summary[
                "reconstruction_source_sha256"
            ],
            "source_capture_summary_sha256": old_summary["summary_sha256"],
            "source_capture_artifact_manifest_sha256": old_artifact[
                "manifest_sha256"
            ],
        }
    )
    _write_artifact_manifest(staging, runtime=runtime)
    _validate_artifact_manifest(staging / "artifact-manifest.json")
    policy_root.parent.mkdir(parents=True, exist_ok=True)
    os.replace(staging, policy_root)
    artifacts.commit()
    return policy_summary


def _capture_records(
    *,
    split: str,
    run_id: str,
    limit: int | None,
    shard_index: int,
    shard_count: int,
    max_completion_tokens: int,
) -> dict[str, Any]:
    corpus_path = Path("/opt/opjax/replay-corpus.json")
    split_path = Path("/opt/opjax/split-manifest.json")
    corpus = json.loads(corpus_path.read_text(encoding="utf-8"))
    split_manifest = json.loads(split_path.read_text(encoding="utf-8"))
    selected = select_final_requests(corpus, split_manifest)[split]
    if limit is not None:
        selected = selected[:limit]
    selected = selected[shard_index::shard_count]
    run_root = ROOT / run_id / "shards" / f"{split}-{shard_index:03d}-of-{shard_count:03d}"
    raw_root = run_root / "raw" / split
    sample_root = run_root / "samples" / split
    profile_root = run_root / "profiles" / split
    raw_root.mkdir(parents=True, exist_ok=True)
    sample_root.mkdir(parents=True, exist_ok=True)
    profile_root.mkdir(parents=True, exist_ok=True)
    command = server_command(
        DFLASH,
        port=PORT,
        proposal_tokens=15,
        capture_target=True,
    )
    _replace_argument(command, "--max-num-seqs", "1")
    command.extend(
        [
            "--enforce-eager",
            "--profiler-config",
            json.dumps(
                {
                    "profiler": "torch",
                    "torch_profiler_dir": str(profile_root),
                    "torch_profiler_with_stack": False,
                    "torch_profiler_record_shapes": True,
                    "torch_profiler_with_memory": True,
                    "torch_profiler_use_gzip": True,
                },
                sort_keys=True,
                separators=(",", ":"),
            ),
        ]
    )
    environment = {
        **os.environ,
        "OPJAX_LAGUNA_TARGET_CAPTURE_ROOT": str(raw_root),
        "OPJAX_SPEC_RUN_ID": f"{run_id}-{split}",
    }
    log_path = run_root / f"server-{split}.log"
    results = []
    started = time.time()
    with log_path.open("wb") as server_log:
        process = subprocess.Popen(
            command,
            env=environment,
            stdout=server_log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            _wait_ready(PORT, process, log_path=log_path)
            for index, record in enumerate(selected):
                session = str(record["prompt_id"])
                session_root = raw_root / session
                sample_path = sample_root / session
                if sample_path.is_dir():
                    try:
                        results.append(validate_sample(sample_path))
                        continue
                    except Exception as error:
                        archive = run_root / "attempts" / f"sample-{session}-{time.time_ns()}"
                        archive.parent.mkdir(parents=True, exist_ok=True)
                        os.replace(sample_path, archive)
                        (archive / "archive.json").write_text(
                            json.dumps({"reason": str(error)}, sort_keys=True) + "\n",
                            encoding="utf-8",
                        )
                if session_root.exists():
                    archive = run_root / "attempts" / f"raw-{session}-{time.time_ns()}"
                    archive.parent.mkdir(parents=True, exist_ok=True)
                    os.replace(session_root, archive)
                (raw_root / "active.json").write_text(
                    json.dumps({"session": session}), encoding="utf-8"
                )
                tokenized = json.loads(
                    _request(
                        f"http://127.0.0.1:{PORT}/tokenize",
                        payload={
                            "model": TARGET_ID,
                            "messages": record["messages"],
                            "tools": [BASH_TOOL],
                            "add_generation_prompt": True,
                            "chat_template_kwargs": {"enable_thinking": True},
                        },
                        timeout=300,
                    )
                )
                prompt_ids = tokenized["tokens"]
                max_tokens = min(max_completion_tokens, 18_432 - len(prompt_ids))
                if max_tokens < 14:
                    raise RuntimeError(
                        f"SERVING_NATIVE_CONTEXT_EXHAUSTED:{session}:{len(prompt_ids)}"
                    )
                if index == 0:
                    _request(
                        f"http://127.0.0.1:{PORT}/start_profile",
                        payload={},
                        timeout=300,
                    )
                response = json.loads(
                    _request(
                        f"http://127.0.0.1:{PORT}/v1/chat/completions",
                        payload=_request_payload(record, max_tokens),
                        timeout=3_600,
                    )
                )
                (session_root / "response.json").write_text(
                    json.dumps(response, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                if index == 0:
                    _request(
                        f"http://127.0.0.1:{PORT}/stop_profile",
                        payload={},
                        timeout=600,
                    )
                (raw_root / "active.json").unlink()
                choice = (response.get("choices") or [{}])[0]
                completion_ids = choice.get("token_ids")
                if not isinstance(completion_ids, list) or not completion_ids:
                    raise RuntimeError(f"SERVING_NATIVE_COMPLETION_INVALID:{session}")
                usage = response.get("usage") or {}
                prompt_token_count = usage.get("prompt_tokens")
                if not isinstance(prompt_token_count, int):
                    raise RuntimeError(f"SERVING_NATIVE_USAGE_INVALID:{session}")
                captured_prompt_ids = captured_prompt_token_ids(
                    session_root=session_root,
                    prompt_token_count=prompt_token_count,
                )
                sample, metadata = reconstruct_committed_sample(
                    session_root=session_root,
                    prompt_token_ids=captured_prompt_ids,
                    completion_token_ids=completion_ids,
                )
                metadata.update(
                    {
                        "prompt_id": record["prompt_id"],
                        "trajectory": record["trajectory"],
                        "call": record["call"],
                        "finish_reason": choice.get("finish_reason"),
                        "tokenize_prompt_tokens": len(prompt_ids),
                        "runtime_prompt_tokens": prompt_token_count,
                        "tokenize_matches_runtime": prompt_ids == captured_prompt_ids,
                    }
                )
                metadata["metadata_sha256"] = canonical_sha256(
                    {
                        key: value
                        for key, value in metadata.items()
                        if key != "metadata_sha256"
                    }
                )
                temporary_sample = sample_root / f".{session}.{time.time_ns()}.tmp"
                manifest = write_sample(temporary_sample, sample=sample, metadata=metadata)
                result = {
                    "prompt_id": session,
                    "trajectory": record["trajectory"],
                    "manifest_sha256": manifest["manifest_sha256"],
                    "completion_tokens": len(completion_ids),
                }
                result["result_sha256"] = canonical_sha256(result)
                (temporary_sample / "capture-result.json").write_text(
                    json.dumps(result, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                validate_sample(temporary_sample)
                os.replace(temporary_sample, sample_path)
                results.append(result)
                artifacts.commit()
        finally:
            (raw_root / "active.json").unlink(missing_ok=True)
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=30)
    summary: dict[str, Any] = {
        "schema_version": 1,
        "kind": "opjax_laguna_serving_native_capture",
        "run_id": run_id,
        "split": split,
        "shard_index": shard_index,
        "shard_count": shard_count,
        "records": results,
        "record_count": len(results),
        "trajectory_count": len({item["trajectory"] for item in results}),
        "source": {
            "replay_corpus_sha256": _sha256(corpus_path),
            "split_manifest_sha256": _sha256(split_path),
        },
        "server_log": {"path": log_path.name, "sha256": _sha256(log_path)},
        "profile": _profile_receipt(profile_root),
        "wall_seconds": time.time() - started,
    }
    summary["summary_sha256"] = canonical_sha256(summary)
    summary_path = run_root / f"capture-{split}.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_artifact_manifest(
        run_root,
        runtime={
            "command": command,
            "split": split,
            "run_id": run_id,
            "target_capture": True,
        },
    )
    artifacts.commit()
    return summary


@app.function(
    image=image,
    gpu="H200",
    volumes={
        HF_CACHE_DIR: cache,
        str(ROOT): artifacts,
        str(TRAINING_ROOT): training,
    },
    secrets=[secret],
    timeout=86_400,
    memory=32_768,
)
def capture_split(
    split: str,
    run_id: str = "serving-native-v1",
    limit: int | None = None,
    shard_index: int = 0,
    shard_count: int = 1,
    max_completion_tokens: int = 8_192,
) -> dict[str, Any]:
    if split not in {"train", "calibration", "heldout"}:
        raise ValueError(f"SERVING_NATIVE_SPLIT_INVALID:{split}")
    if limit is not None and limit < 1:
        raise ValueError(f"SERVING_NATIVE_LIMIT_INVALID:{limit}")
    if shard_count < 1 or not 0 <= shard_index < shard_count:
        raise ValueError(
            f"SERVING_NATIVE_SHARD_INVALID:{shard_index}:{shard_count}"
        )
    if not 14 <= max_completion_tokens <= 8_192:
        raise ValueError(
            f"SERVING_NATIVE_MAX_COMPLETION_INVALID:{max_completion_tokens}"
        )
    return _capture_records(
        split=split,
        run_id=run_id,
        limit=limit,
        shard_index=shard_index,
        shard_count=shard_count,
        max_completion_tokens=max_completion_tokens,
    )


@app.function(
    image=image,
    gpu="H200",
    volumes={HF_CACHE_DIR: cache, str(ROOT): artifacts},
    secrets=[secret],
    timeout=86_400,
    memory=32_768,
)
def capture_fixed_split(
    split: str,
    run_id: str = "serving-native-v2",
    shard_index: int = 0,
    shard_count: int = 1,
) -> dict[str, Any]:
    if split not in {"train", "calibration", "heldout"}:
        raise ValueError(f"SERVING_NATIVE_SPLIT_INVALID:{split}")
    if shard_count < 1 or not 0 <= shard_index < shard_count:
        raise ValueError(f"SERVING_NATIVE_SHARD_INVALID:{shard_index}:{shard_count}")
    return _capture_fixed_records(
        split=split,
        run_id=run_id,
        shard_index=shard_index,
        shard_count=shard_count,
    )


@app.function(
    image=image,
    volumes={str(ROOT): artifacts},
    timeout=86_400,
    memory=32_768,
)
def rebuild_fixed_split(
    split: str,
    run_id: str = "serving-native-v2",
    shard_index: int = 0,
    shard_count: int = 1,
) -> dict[str, Any]:
    if split not in {"train", "calibration", "heldout"}:
        raise ValueError(f"SERVING_NATIVE_SPLIT_INVALID:{split}")
    if shard_count < 1 or not 0 <= shard_index < shard_count:
        raise ValueError(f"SERVING_NATIVE_SHARD_INVALID:{shard_index}:{shard_count}")
    return _rebuild_fixed_records(
        split=split,
        run_id=run_id,
        shard_index=shard_index,
        shard_count=shard_count,
    )


@app.function(image=image, volumes={str(ROOT): artifacts}, timeout=3600, memory=8192)
def aggregate_capture(
    run_id: str,
    train_shards: int,
    eval_shards: int,
) -> dict[str, Any]:
    run_root = ROOT / run_id
    expected = {"train": train_shards, "calibration": eval_shards, "heldout": eval_shards}
    corpus = json.loads(Path("/opt/opjax/replay-corpus.json").read_text(encoding="utf-8"))
    split_manifest = json.loads(
        Path("/opt/opjax/split-manifest.json").read_text(encoding="utf-8")
    )
    selected = select_final_requests(corpus, split_manifest)
    split_results: dict[str, Any] = {}
    all_prompt_ids: set[str] = set()
    for split, shard_count in expected.items():
        summaries = []
        records = []
        for index in range(shard_count):
            shard_root = run_root / "shards" / f"{split}-{index:03d}-of-{shard_count:03d}"
            summary_path = shard_root / f"capture-{split}.json"
            artifact_path = shard_root / "artifact-manifest.json"
            if not summary_path.is_file() or not artifact_path.is_file():
                raise RuntimeError(f"SERVING_NATIVE_SHARD_INCOMPLETE:{shard_root}")
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            if canonical_sha256(
                {key: value for key, value in summary.items() if key != "summary_sha256"}
            ) != summary.get("summary_sha256"):
                raise RuntimeError(f"SERVING_NATIVE_SUMMARY_HASH_INVALID:{shard_root}")
            artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
            if canonical_sha256(
                {key: value for key, value in artifact.items() if key != "manifest_sha256"}
            ) != artifact.get("manifest_sha256"):
                raise RuntimeError(f"SERVING_NATIVE_ARTIFACT_HASH_INVALID:{shard_root}")
            summaries.append(
                {
                    "path": str(summary_path.relative_to(run_root)),
                    "sha256": _sha256(summary_path),
                    "summary_sha256": summary["summary_sha256"],
                    "artifact_manifest_sha256": artifact["manifest_sha256"],
                }
            )
            records.extend(summary["records"])
            for record in summary["records"]:
                sample = shard_root / "samples" / split / record["prompt_id"]
                validate_sample(sample)
        actual_ids = [record["prompt_id"] for record in records]
        expected_ids = [record["prompt_id"] for record in selected[split]]
        if len(actual_ids) != len(set(actual_ids)) or set(actual_ids) != set(expected_ids):
            raise RuntimeError(
                f"SERVING_NATIVE_AGGREGATE_IDS_INVALID:{split}:"
                f"{len(actual_ids)}:{len(expected_ids)}"
            )
        if all_prompt_ids & set(actual_ids):
            raise RuntimeError(f"SERVING_NATIVE_AGGREGATE_LEAKAGE:{split}")
        all_prompt_ids.update(actual_ids)
        split_results[split] = {
            "record_count": len(records),
            "shards": summaries,
            "prompt_ids_sha256": hashlib.sha256(
                "\n".join(sorted(actual_ids)).encode()
            ).hexdigest(),
        }
    result: dict[str, Any] = {
        "schema_version": 1,
        "kind": "opjax_laguna_serving_native_capture_release",
        "run_id": run_id,
        "splits": split_results,
        "source": {
            "replay_corpus_sha256": _sha256(Path("/opt/opjax/replay-corpus.json")),
            "split_manifest_sha256": _sha256(Path("/opt/opjax/split-manifest.json")),
        },
    }
    result["release_sha256"] = canonical_sha256(result)
    (run_root / "release.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    artifacts.commit()
    return result


@app.function(image=image, volumes={str(ROOT): artifacts}, timeout=7200, memory=16_384)
def aggregate_fixed_capture(
    run_id: str,
    train_shards: int,
    eval_shards: int,
) -> dict[str, Any]:
    run_root = ROOT / run_id
    token_release_path = run_root / "tokenized" / "release.json"
    token_release = json.loads(token_release_path.read_text(encoding="utf-8"))
    if canonical_sha256(
        {key: value for key, value in token_release.items() if key != "release_sha256"}
    ) != token_release.get("release_sha256"):
        raise RuntimeError("SERVING_NATIVE_TOKEN_RELEASE_HASH_INVALID")
    split_results = {}
    all_ids: set[str] = set()
    for split, shard_count in (
        ("train", train_shards),
        ("calibration", eval_shards),
        ("heldout", eval_shards),
    ):
        summaries = []
        records = []
        for index in range(shard_count):
            shard_root = (
                run_root
                / "rebuilt-fixed-shards"
                / f"{split}-{index:03d}-of-{shard_count:03d}"
                / "policies"
                / "first-causal-observation-wins"
            )
            summary_path = shard_root / "policy-summary.json"
            artifact_path = shard_root / "artifact-manifest.json"
            artifact = _validate_artifact_manifest(artifact_path)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            if canonical_sha256(
                {key: value for key, value in summary.items() if key != "summary_sha256"}
            ) != summary.get("summary_sha256"):
                raise RuntimeError(f"SERVING_NATIVE_SUMMARY_HASH_INVALID:{summary_path}")
            if (
                summary.get("token_release_sha256") != token_release["release_sha256"]
                or summary.get("shard_index") != index
                or summary.get("shard_count") != shard_count
                or summary.get("split") != split
                or summary.get("profile", {}).get("cuda_launch_events", 0) < 1
                or summary.get("feature_selection_policy")
                != "first_causal_observation_wins"
                or summary.get("rebuild_driver_sha256") != _sha256(Path(__file__))
                or summary.get("reconstruction_source_sha256")
                != _sha256(Path(write_sample.__code__.co_filename))
                or artifact.get("runtime", {}).get("rebuild_driver_sha256")
                != summary.get("rebuild_driver_sha256")
                or artifact.get("runtime", {}).get(
                    "reconstruction_source_sha256"
                )
                != summary.get("reconstruction_source_sha256")
            ):
                raise RuntimeError(f"SERVING_NATIVE_SUMMARY_CONTRACT:{summary_path}")
            runtime_record = summary.get("runtime", {})
            runtime_path = ROOT / str(runtime_record.get("path"))
            if (
                not runtime_path.is_file()
                or _sha256(runtime_path) != runtime_record.get("sha256")
            ):
                raise RuntimeError(f"SERVING_NATIVE_RUNTIME_ARTIFACT:{summary_path}")
            summaries.append(
                {
                    "path": str(summary_path.relative_to(run_root)),
                    "file_sha256": _sha256(summary_path),
                    "summary_sha256": summary["summary_sha256"],
                    "artifact_manifest_sha256": artifact["manifest_sha256"],
                }
            )
            records.extend(summary["records"])
            for record in summary["records"]:
                sample_root = shard_root / "samples" / split / record["prompt_id"]
                validated = validate_sample(sample_root)
                sample_manifest = json.loads(
                    (sample_root / "manifest.json").read_text(encoding="utf-8")
                )
                if (
                    validated != record
                    or _sha256(sample_root / "manifest.json")
                    != record.get("manifest_file_sha256")
                    or sample_manifest.get("metadata", {}).get(
                        "feature_selection_policy"
                    )
                    != "first_causal_observation_wins"
                    or sample_manifest.get("metadata", {}).get(
                        "rebuild_driver_sha256"
                    )
                    != summary.get("rebuild_driver_sha256")
                    or sample_manifest.get("metadata", {}).get(
                        "reconstruction_source_sha256"
                    )
                    != summary.get("reconstruction_source_sha256")
                    or sample_manifest.get("metadata", {}).get(
                        "source_capture_artifact_manifest_sha256"
                    )
                    != summary.get("source_capture_artifact_manifest_sha256")
                ):
                    raise RuntimeError(
                        f"SERVING_NATIVE_RELEASE_SAMPLE_INVALID:{split}:"
                        f"{record['prompt_id']}"
                    )
        actual_ids = [record["prompt_id"] for record in records]
        expected_records = token_release["splits"][split]["records"]
        expected_ids = [record["id"] for record in expected_records]
        if len(actual_ids) != len(set(actual_ids)) or set(actual_ids) != set(expected_ids):
            raise RuntimeError(f"SERVING_NATIVE_RELEASE_IDS_INVALID:{split}")
        if all_ids & set(actual_ids):
            raise RuntimeError(f"SERVING_NATIVE_RELEASE_SPLIT_LEAKAGE:{split}")
        all_ids.update(actual_ids)
        expected_manifests = {
            record["id"]: record["manifest_sha256"] for record in expected_records
        }
        if any(
            record.get("token_manifest_sha256")
            != expected_manifests[record["prompt_id"]]
            for record in records
        ):
            raise RuntimeError(f"SERVING_NATIVE_RELEASE_TOKEN_BINDING:{split}")
        split_results[split] = {
            "record_count": len(records),
            "records": [
                {
                    "id": record["prompt_id"],
                    "trajectory": record["trajectory"],
                    "manifest_sha256": record["manifest_sha256"],
                    "manifest_file_sha256": record["manifest_file_sha256"],
                    "token_manifest_sha256": record["token_manifest_sha256"],
                    "tokens": record["tokens"],
                }
                for record in sorted(records, key=lambda item: item["prompt_id"])
            ],
            "records_sha256": canonical_sha256(
                {"records": sorted(records, key=lambda item: item["prompt_id"])}
            ),
            "feature_selection_policy": "first_causal_observation_wins",
            "rebuild_driver_sha256": _sha256(Path(__file__)),
            "reconstruction_source_sha256": _sha256(
                Path(write_sample.__code__.co_filename)
            ),
            "shards": summaries,
        }
    release: dict[str, Any] = {
        "schema_version": 1,
        "kind": "opjax_laguna_serving_native_fixed_release",
        "run_id": run_id,
        "token_release": {
            "path": str(token_release_path.relative_to(run_root)),
            "file_sha256": _sha256(token_release_path),
            "release_sha256": token_release["release_sha256"],
        },
        "splits": split_results,
    }
    release["release_sha256"] = canonical_sha256(release)
    (run_root / "release.json").write_text(
        json.dumps(release, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    artifacts.commit()
    return release


@app.local_entrypoint()
def main(
    run_id: str = "serving-native-v2",
    train_shards: int = 12,
    eval_shards: int = 3,
) -> None:
    jobs = []
    for split, shard_count in (
        ("train", train_shards),
        ("calibration", eval_shards),
        ("heldout", eval_shards),
    ):
        jobs.extend(
            (split, run_id, index, shard_count)
            for index in range(shard_count)
        )
    results = list(capture_fixed_split.starmap(jobs, order_outputs=False))
    if len(results) != len(jobs):
        raise RuntimeError(f"SERVING_NATIVE_CAPTURE_JOB_COUNT:{len(results)}:{len(jobs)}")
    print(
        json.dumps(
            aggregate_fixed_capture.remote(run_id, train_shards, eval_shards),
            indent=2,
            sort_keys=True,
        )
    )


@app.local_entrypoint()
def rebuild_main(
    run_id: str = "serving-native-v2",
    train_shards: int = 12,
    eval_shards: int = 3,
) -> None:
    jobs = [
        (split, run_id, index, shard_count)
        for split, shard_count in (
            ("train", train_shards),
            ("calibration", eval_shards),
            ("heldout", eval_shards),
        )
        for index in range(shard_count)
    ]
    results = list(rebuild_fixed_split.starmap(jobs, order_outputs=False))
    if len(results) != len(jobs):
        raise RuntimeError(f"SERVING_NATIVE_REBUILD_JOB_COUNT:{len(results)}:{len(jobs)}")
    print(
        json.dumps(
            aggregate_fixed_capture.remote(run_id, train_shards, eval_shards),
            indent=2,
            sort_keys=True,
        )
    )
