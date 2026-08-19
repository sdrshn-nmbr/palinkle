from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import urllib.error
import urllib.request

from opjax.pallas.laguna_speculative import (
    DFLASH,
    DSPARK,
    PLAIN,
    TARGET_REVISION,
    VLLM_IMAGE,
    canonical_sha256,
    server_command,
)


METADATA_ROOT = "http://metadata.google.internal/computeMetadata/v1"
TOKENIZER_FILES = (
    "chat_template.jinja",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
)


def _gce_metadata(path: str) -> str:
    request = urllib.request.Request(
        f"{METADATA_ROOT}/{path}", headers={"Metadata-Flavor": "Google"}
    )
    with urllib.request.urlopen(request, timeout=10) as response:
        value = response.read().decode().strip()
    if not value:
        raise RuntimeError(f"LAGUNA_GCE_METADATA_EMPTY:{path}")
    return value


def _source_hashes(source_root: Path) -> dict[str, str]:
    paths = (
        Path(__file__),
        Path(__file__).with_name("laguna_trained_replay.py"),
        source_root / "opjax" / "remote" / "laguna_vllm_entrypoint.py",
        source_root / "opjax" / "remote" / "gpu_runtime_identity.py",
        source_root / "opjax" / "pallas" / "laguna_speculative.py",
    )
    return {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths
    }


def _tokenizer_receipt(hf_root: Path) -> dict[str, object]:
    relative = (
        Path("hub")
        / "models--poolside--Laguna-XS-2.1"
        / "snapshots"
        / TARGET_REVISION
    )
    snapshot = hf_root / relative
    files = {}
    for name in TOKENIZER_FILES:
        path = snapshot / name
        raw = path.read_bytes()
        if not raw:
            raise RuntimeError(f"LAGUNA_TOKENIZER_FILE_EMPTY:{path}")
        files[name] = {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
    return {
        "revision": TARGET_REVISION,
        "container_path": f"/hf/{relative}",
        "files": files,
    }


def _resolve_container_interpreter() -> dict[str, str]:
    command = (
        "sudo",
        "docker",
        "run",
        "--rm",
        "--entrypoint",
        "/bin/sh",
        VLLM_IMAGE,
        "-lc",
        "set -eu; launcher=$(command -v vllm); head -1 \"$launcher\"; "
        "printf '%s\\n' \"$launcher\"",
    )
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    lines = result.stdout.splitlines()
    if len(lines) != 2 or not lines[0].startswith("#!"):
        raise RuntimeError(f"LAGUNA_CONTAINER_LAUNCHER_INVALID:{lines}")
    interpreter = lines[0][2:]
    probe = subprocess.run(
        (
            "sudo",
            "docker",
            "run",
            "--rm",
            "--entrypoint",
            interpreter,
            VLLM_IMAGE,
            "-c",
            "import sys; print(sys.executable)",
        ),
        check=True,
        capture_output=True,
        text=True,
    )
    observed = probe.stdout.strip()
    if observed != interpreter:
        raise RuntimeError(
            f"LAGUNA_CONTAINER_INTERPRETER_MISMATCH:{interpreter}:{observed}"
        )
    return {"interpreter": interpreter, "vllm_launcher": lines[1]}


def _write_attempt_receipt(
    *,
    path: Path,
    attempt_id: str,
    declared_gpu: str,
    gpu_count: int,
    tensor_parallel_size: int,
    gpu_memory_utilization: float,
    kv_cache_memory_bytes: int,
    disable_custom_all_reduce: bool,
    deployment_id: str,
    deployment_zone: str,
    instance_id: str,
    source_root: Path,
    tokenizer: dict[str, object],
    container_launcher: dict[str, str],
) -> None:
    receipt = {
        "schema_version": 1,
        "kind": "opjax_laguna_gce_replay_attempt",
        "attempt_id": attempt_id,
        "declared_gpu": declared_gpu,
        "gpu_count": gpu_count,
        "tensor_parallel_size": tensor_parallel_size,
        "gpu_memory_utilization": gpu_memory_utilization,
        "kv_cache_memory_bytes": kv_cache_memory_bytes,
        "disable_custom_all_reduce": disable_custom_all_reduce,
        "deployment": {
            "provider": "gcp_compute_engine",
            "id": deployment_id,
            "zone": deployment_zone,
            "instance_id": instance_id,
        },
        "image": VLLM_IMAGE,
        "container_launcher": container_launcher,
        "tokenizer": tokenizer,
        "measurement_sources": _source_hashes(source_root),
    }
    receipt["sha256"] = canonical_sha256(receipt)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(receipt, indent=2, sort_keys=True) + "\n")


def _file_receipt(path: Path, relative_path: str) -> dict[str, str | int]:
    raw = path.read_bytes()
    if not raw:
        raise RuntimeError(f"LAGUNA_CELL_ARTIFACT_EMPTY:{path}")
    return {
        "path": relative_path,
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def _bind_cell_artifacts(
    *,
    cell: str,
    attempt_id: str,
    output_root: Path,
    artifact_root: Path,
    pre_stop_returncode: int | None,
    docker_stop_returncode: int,
    process_returncode: int,
    tokenizer: dict[str, object],
) -> None:
    arm, depth = _cell(cell)
    runtime_relative = Path(arm) / _run_id(attempt_id, arm, depth) / "runtime.json"
    gpu_relative = runtime_relative.with_name("gpu.csv")
    log_relative = Path(f"{cell}.server.log")
    payload = {
        "schema_version": 1,
        "kind": "opjax_laguna_replay_cell_artifacts",
        "cell": cell,
        "attempt_id": attempt_id,
        "measurement_completed": True,
        "server_termination": "stopped_after_measurement",
        "pre_stop_returncode": pre_stop_returncode,
        "docker_stop_returncode": docker_stop_returncode,
        "server_process_returncode": process_returncode,
        "tokenizer": tokenizer,
        "files": {
            "server_log": _file_receipt(
                output_root / log_relative, str(log_relative)
            ),
            "gpu_csv": _file_receipt(
                artifact_root / gpu_relative, str(gpu_relative)
            ),
            "runtime": _file_receipt(
                artifact_root / runtime_relative, str(runtime_relative)
            ),
        },
    }
    payload["sha256"] = canonical_sha256(payload)
    receipt_path = output_root / f"{cell}.artifact.json"
    with receipt_path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    result_path = output_root / f"{cell}.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    expected_result = canonical_sha256(
        {key: value for key, value in result.items() if key != "result_sha256"}
    )
    if result.get("result_sha256") != expected_result:
        raise RuntimeError(f"LAGUNA_CELL_RESULT_HASH_MISMATCH:{cell}")
    result["cell_artifact_receipt"] = {
        "sha256": payload["sha256"],
        "file_sha256": hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
        "payload": payload,
    }
    result["result_sha256"] = canonical_sha256(
        {key: value for key, value in result.items() if key != "result_sha256"}
    )
    temporary = result_path.with_suffix(".json.pending")
    temporary.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, result_path)


def _cell(cell: str) -> tuple[str, int | None]:
    if cell == PLAIN:
        return PLAIN, None
    arm, depth = cell.split("-", maxsplit=1)
    if arm not in {DFLASH, DSPARK} or not depth.isdigit():
        raise ValueError(f"LAGUNA_DOCKER_CELL_INVALID:{cell}")
    return arm, int(depth)


def _run_id(attempt_id: str, arm: str, depth: int | None) -> str:
    if arm == PLAIN:
        return f"{attempt_id}-released-plain"
    return f"{attempt_id}-trained-{arm}-fixed-{depth}"


def _wait_ready(process: subprocess.Popen[bytes], *, timeout: int) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                f"LAGUNA_DOCKER_SERVER_EXITED:{process.returncode}"
            )
        try:
            with urllib.request.urlopen(
                "http://127.0.0.1:8000/health", timeout=5
            ) as response:
                if response.status == 200:
                    return
        except (OSError, urllib.error.URLError):
            pass
        time.sleep(5)
    raise RuntimeError("LAGUNA_DOCKER_SERVER_READY_TIMEOUT")


def _docker_command(
    *,
    cell: str,
    attempt_id: str,
    declared_gpu: str,
    source_root: Path,
    checkpoint_root: Path,
    artifact_root: Path,
    hf_root: Path,
    deployment_id: str,
    deployment_zone: str,
    deployment_instance_id: str,
    container_interpreter: str,
    tensor_parallel_size: int,
    gpu_memory_utilization: float,
    kv_cache_memory_bytes: int,
    tokenizer_path: str,
    disable_custom_all_reduce: bool,
) -> tuple[list[str], str]:
    arm, depth = _cell(cell)
    draft_model = (
        None if arm == PLAIN else f"/checkpoints/{arm}/step_120"
    )
    serve = server_command(
        arm,
        port=8000,
        proposal_tokens=depth,
        adaptive_verification=(
            False if arm == DSPARK and depth is not None else None
        ),
        draft_model=draft_model,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        kv_cache_memory_bytes=kv_cache_memory_bytes,
        tokenizer=tokenizer_path,
        disable_custom_all_reduce=disable_custom_all_reduce,
    )
    container_name = f"opjax-laguna-{cell.replace('_', '-') }"
    command = [
        "sudo",
        "docker",
        "run",
        "--rm",
        "--name",
        container_name,
        "--gpus",
        "all",
        "--network",
        "host",
        "--ipc",
        "host",
        "-v",
        f"{source_root}:/workspace/src:ro",
        "-v",
        f"{checkpoint_root}:/checkpoints:ro",
        "-v",
        f"{artifact_root}:/artifacts",
        "-v",
        f"{hf_root}:/hf",
        "-e",
        "PYTHONPATH=/workspace/src",
        "-e",
        "HF_HOME=/hf",
        "-e",
        "HF_HUB_CACHE=/hf/hub",
        "-e",
        "HF_HUB_ENABLE_HF_TRANSFER=1",
        "-e",
        "OPJAX_SPEC_ARTIFACT_ROOT=/artifacts",
        "-e",
        f"OPJAX_SPEC_ATTEMPT_ID={attempt_id}",
        "-e",
        f"OPJAX_SPEC_DECLARED_GPU={declared_gpu}",
        "-e",
        f"OPJAX_EXPECTED_GPU_COUNT={tensor_parallel_size}",
        "-e",
        f"OPJAX_SPEC_RUN_ID={_run_id(attempt_id, arm, depth)}",
        "-e",
        "OPJAX_DEPLOYMENT_PROVIDER=gcp_compute_engine",
        "-e",
        f"OPJAX_DEPLOYMENT_ID={deployment_id}",
        "-e",
        f"OPJAX_DEPLOYMENT_ZONE={deployment_zone}",
        "-e",
        f"OPJAX_DEPLOYMENT_INSTANCE_ID={deployment_instance_id}",
        "-e",
        f"OPJAX_CONTAINER_INTERPRETER={container_interpreter}",
        "--entrypoint",
        container_interpreter,
        VLLM_IMAGE,
        *serve[1:],
    ]
    return command, container_name


def _replay_client_command(
    *,
    container_interpreter: str,
    source_root: Path,
    scripts_root: Path,
    selection_root: Path,
    output_root: Path,
    artifact_root: Path,
    corpus: Path,
    corpus_manifest: Path,
    split: str,
    cells: str,
    attempt_id: str,
    declared_gpu: str,
    gpu_count: int,
    tensor_parallel_size: int,
    gpu_memory_utilization: float,
    kv_cache_memory_bytes: int,
    disable_custom_all_reduce: bool,
    endpoint: str | None,
    defer_summary: bool,
    finalize_existing: bool,
) -> tuple[str, ...]:
    command = [
        "sudo",
        "docker",
        "run",
        "--rm",
        "--network",
        "host",
        "-v",
        f"{source_root}:/workspace/src:ro",
        "-v",
        f"{scripts_root}:/workspace/scripts/pallas:ro",
        "-v",
        f"{selection_root}:/selection:ro",
        "-v",
        f"{output_root}:/output",
        "-v",
        f"{artifact_root}:/artifacts:ro",
        "-v",
        f"{corpus}:/inputs/replay-corpus.json:ro",
        "-v",
        f"{corpus_manifest}:/inputs/corpus-manifest.json:ro",
        "-e",
        "PYTHONPATH=/workspace/src",
        "--entrypoint",
        container_interpreter,
        VLLM_IMAGE,
        "/workspace/scripts/pallas/laguna_trained_replay.py",
        "--split",
        split,
        "--cells",
        cells,
        "--corpus",
        "/inputs/replay-corpus.json",
        "--corpus-manifest",
        "/inputs/corpus-manifest.json",
        "--selection-root",
        "/selection",
        "--output-root",
        "/output",
        "--attempt-id",
        attempt_id,
        "--expected-gpu-name",
        declared_gpu,
        "--expected-gpu-count",
        str(gpu_count),
        "--expected-tensor-parallel-size",
        str(tensor_parallel_size),
        "--expected-gpu-memory-utilization",
        str(gpu_memory_utilization),
        "--expected-kv-cache-memory-bytes",
        str(kv_cache_memory_bytes),
        "--runtime-root",
        "/artifacts",
        "--attempt-receipt",
        "/output/attempt.json",
    ]
    if disable_custom_all_reduce:
        command.append("--expected-custom-all-reduce-disabled")
    if endpoint is not None:
        command.extend(("--endpoint", endpoint))
    if defer_summary:
        command.extend(("--defer-summary", "--max-workers", "1"))
    if finalize_existing:
        command.append("--finalize-existing")
    return tuple(command)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("calibration", "heldout"), required=True)
    parser.add_argument("--cells", required=True)
    parser.add_argument("--attempt-id", required=True)
    parser.add_argument("--declared-gpu", required=True)
    parser.add_argument("--gpu-count", type=int, default=1)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--deployment-id", required=True)
    parser.add_argument("--deployment-zone", required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--hf-root", type=Path, required=True)
    parser.add_argument("--selection-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--corpus-manifest", type=Path, required=True)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--kv-cache-memory-bytes", type=int, required=True)
    parser.add_argument("--disable-custom-all-reduce", action="store_true")
    args = parser.parse_args()
    cells = [cell for cell in args.cells.split(",") if cell]
    if args.gpu_count < 1 or args.tensor_parallel_size != args.gpu_count:
        raise ValueError("LAGUNA_DOCKER_GPU_TOPOLOGY_INVALID")
    if not 0.0 < args.gpu_memory_utilization < 1.0:
        raise ValueError("LAGUNA_DOCKER_GPU_MEMORY_UTILIZATION_INVALID")
    if args.kv_cache_memory_bytes < 1:
        raise ValueError("LAGUNA_DOCKER_KV_CACHE_MEMORY_BYTES_INVALID")
    if PLAIN not in cells:
        raise ValueError("LAGUNA_DOCKER_PLAIN_REQUIRED")
    for path in (
        args.source_root,
        args.checkpoint_root,
        args.selection_root,
    ):
        if not path.exists():
            raise ValueError(f"LAGUNA_DOCKER_INPUT_MISSING:{path}")
    args.artifact_root.mkdir(parents=True, exist_ok=True)
    args.hf_root.mkdir(parents=True, exist_ok=True)
    args.output_root.mkdir(parents=True, exist_ok=False)
    tokenizer = _tokenizer_receipt(args.hf_root)
    subprocess.run(
        ("sudo", "docker", "pull", VLLM_IMAGE),
        check=True,
    )
    container_launcher = _resolve_container_interpreter()
    instance_id = _gce_metadata("instance/id")
    observed_zone = _gce_metadata("instance/zone").rsplit("/", maxsplit=1)[-1]
    if observed_zone != args.deployment_zone:
        raise ValueError(
            f"LAGUNA_GCE_ZONE_MISMATCH:{args.deployment_zone}:{observed_zone}"
        )
    _write_attempt_receipt(
        path=args.output_root / "attempt.json",
        attempt_id=args.attempt_id,
        declared_gpu=args.declared_gpu,
        gpu_count=args.gpu_count,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        kv_cache_memory_bytes=args.kv_cache_memory_bytes,
        disable_custom_all_reduce=args.disable_custom_all_reduce,
        deployment_id=args.deployment_id,
        deployment_zone=args.deployment_zone,
        instance_id=instance_id,
        source_root=args.source_root,
        tokenizer=tokenizer,
        container_launcher=container_launcher,
    )
    for cell in cells:
        docker_command, container_name = _docker_command(
            cell=cell,
            attempt_id=args.attempt_id,
            declared_gpu=args.declared_gpu,
            source_root=args.source_root,
            checkpoint_root=args.checkpoint_root,
            artifact_root=args.artifact_root,
            hf_root=args.hf_root,
            deployment_id=args.deployment_id,
            deployment_zone=args.deployment_zone,
            deployment_instance_id=instance_id,
            container_interpreter=container_launcher["interpreter"],
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            kv_cache_memory_bytes=args.kv_cache_memory_bytes,
            tokenizer_path=str(tokenizer["container_path"]),
            disable_custom_all_reduce=args.disable_custom_all_reduce,
        )
        log_path = args.output_root / f"{cell}.server.log"
        measurement_completed = False
        pre_stop_returncode = None
        docker_stop_returncode = -1
        with log_path.open("xb") as log:
            process = subprocess.Popen(
                docker_command,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            try:
                _wait_ready(process, timeout=3600)
                subprocess.run(
                    _replay_client_command(
                        container_interpreter=container_launcher["interpreter"],
                        source_root=args.source_root,
                        scripts_root=Path(__file__).parent,
                        selection_root=args.selection_root,
                        output_root=args.output_root,
                        artifact_root=args.artifact_root,
                        corpus=args.corpus,
                        corpus_manifest=args.corpus_manifest,
                        split=args.split,
                        cells=cell,
                        attempt_id=args.attempt_id,
                        declared_gpu=args.declared_gpu,
                        gpu_count=args.gpu_count,
                        tensor_parallel_size=args.tensor_parallel_size,
                        gpu_memory_utilization=args.gpu_memory_utilization,
                        kv_cache_memory_bytes=args.kv_cache_memory_bytes,
                        disable_custom_all_reduce=args.disable_custom_all_reduce,
                        endpoint="http://127.0.0.1:8000",
                        defer_summary=True,
                        finalize_existing=False,
                    ),
                    check=True,
                )
                measurement_completed = True
            finally:
                pre_stop_returncode = process.poll()
                stop_result = subprocess.run(
                    ("sudo", "docker", "stop", "--timeout", "30", container_name),
                    check=False,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                docker_stop_returncode = stop_result.returncode
                process_returncode = process.wait(timeout=60)
        if not measurement_completed:
            raise RuntimeError(f"LAGUNA_CELL_MEASUREMENT_INCOMPLETE:{cell}")
        if (
            pre_stop_returncode is not None
            or docker_stop_returncode != 0
            or process_returncode not in {0, 137, 143}
        ):
            raise RuntimeError(
                "LAGUNA_CELL_SERVER_TERMINATION_INVALID:"
                f"{cell}:{pre_stop_returncode}:"
                f"{docker_stop_returncode}:{process_returncode}"
            )
        observed_tokenizer = _tokenizer_receipt(args.hf_root)
        if observed_tokenizer != tokenizer:
            raise RuntimeError(f"LAGUNA_TOKENIZER_DRIFT:{cell}")
        _bind_cell_artifacts(
            cell=cell,
            attempt_id=args.attempt_id,
            output_root=args.output_root,
            artifact_root=args.artifact_root,
            pre_stop_returncode=pre_stop_returncode,
            docker_stop_returncode=docker_stop_returncode,
            process_returncode=process_returncode,
            tokenizer=observed_tokenizer,
        )
    if _tokenizer_receipt(args.hf_root) != tokenizer:
        raise RuntimeError("LAGUNA_TOKENIZER_DRIFT:finalize")
    subprocess.run(
        _replay_client_command(
            container_interpreter=container_launcher["interpreter"],
            source_root=args.source_root,
            scripts_root=Path(__file__).parent,
            selection_root=args.selection_root,
            output_root=args.output_root,
            artifact_root=args.artifact_root,
            corpus=args.corpus,
            corpus_manifest=args.corpus_manifest,
            split=args.split,
            cells=",".join(cells),
            attempt_id=args.attempt_id,
            declared_gpu=args.declared_gpu,
            gpu_count=args.gpu_count,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            kv_cache_memory_bytes=args.kv_cache_memory_bytes,
            disable_custom_all_reduce=args.disable_custom_all_reduce,
            endpoint=None,
            defer_summary=False,
            finalize_existing=True,
        ),
        check=True,
    )


if __name__ == "__main__":
    main()
