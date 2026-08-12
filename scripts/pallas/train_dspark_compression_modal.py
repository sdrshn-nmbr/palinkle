from __future__ import annotations

import json
import os
import platform
import signal
import shutil
import socket
import subprocess
import sys
import time
import uuid
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path

import modal


APP_NAME = "opjax-dspark-compression-training"
BASE_IMAGE = "lmsysorg/sglang@sha256:b90c0d760a65bc4dbbe4520bea966c437cc40391dcb7cca2a74922985dc1abeb"
SPECFORGE_REVISION = "e6440f09a8574b35f894608559fd3d165971e488"
TARGET_MODEL = "thinkingmachines/Inkling-Small-NVFP4"
TARGET_REVISION = "b6a99534467840620d411e4cd4ad5819b2610d9c"
INDUCTOR_COMPILE_THREADS = 8
GPU_QUERY_FIELDS = (
    "index",
    "uuid",
    "name",
    "pstate",
    "memory.used",
    "memory.total",
    "utilization.gpu",
    "utilization.memory",
    "power.draw",
    "power.limit",
    "temperature.gpu",
    "clocks.sm",
    "clocks.mem",
)

app = modal.App(APP_NAME)
hf_cache = modal.Volume.from_name(
    "opjax-hf-cache-v2",
    environment_name="main",
    create_if_missing=True,
    version=2,
)
artifacts = modal.Volume.from_name(
    "opjax-dspark-compression-artifacts-20260810",
    environment_name="main",
    create_if_missing=False,
)
secret = modal.Secret.from_name("opjax-secrets", environment_name="main")
image = (
    modal.Image.from_registry(BASE_IMAGE)
    .run_commands(
        "git clone --filter=blob:none https://github.com/sgl-project/SpecForge.git /opt/specforge",
        f"git -C /opt/specforge checkout --detach {SPECFORGE_REVISION}",
        "python -m pip install -e /opt/specforge --no-deps",
        "python -m pip install mooncake-transfer-engine-cuda13",
    )
    .env(
        {
            "HF_XET_HIGH_PERFORMANCE": "1",
            "SGLANG_ENABLE_UNIFIED_RADIX_TREE": "1",
            "SGLANG_OPT_USE_INKLING_CUSTOM_AR": "1",
            "PYTHONPATH": "/opt/specforge",
        }
    )
)


def write_config(
    student: str,
    run_root: Path,
    max_steps: int,
    *,
    trainer_nproc: int = 2,
    accumulation_steps: int = 32,
    moe_backend: str = "flashinfer_trtllm_routed",
    attention_backend: str = "fa4",
    server_mem_fraction: float = 0.60,
) -> Path:
    student_path = Path("/artifacts/students") / student
    config = f"""
model:
  target_model_path: {TARGET_MODEL}
  draft_model_config: {student_path}/config.json
  draft_checkpoint_path: {student_path}
  target_backend: sglang
  trust_remote_code: true
  tokenizer_pad_token_id: 200006
  embedding_key: model.llm.embed.weight
  lm_head_key: model.llm.unembed.weight
  sglang_attention_backend: {attention_backend}
  sglang_mem_fraction_static: {server_mem_fraction}
  sglang_disable_radix_cache: false
  sglang_context_length: 4096
  sglang_moe_runner_backend: {moe_backend}
  sglang_page_size: 128
  sglang_quantization: modelopt_fp4
  sglang_mamba_radix_cache_strategy: extra_buffer
  sglang_max_mamba_cache_size: 64
  sglang_swa_full_tokens_ratio: 0.1
data:
  train_data_path: /artifacts/data/train.jsonl
  max_length: 4096
  chat_template: inkling-thinking
  cache_dir: /artifacts/cache
  build_dataset_num_proc: 4
training:
  strategy: dspark
  num_epochs: 1
  max_steps: {max_steps}
  batch_size: 1
  accumulation_steps: {accumulation_steps}
  learning_rate: 0.0001
  warmup_ratio: 0.04
  max_grad_norm: 1.0
  attention_backend: eager
  num_anchors: 128
  loss_decay_gamma: 4.0
  objective_chunk_blocks: 64
  save_interval: 0
  log_interval: 1
  dist_timeout: 30
  seed: 42
profiling:
  enabled: true
  start_step: {0 if max_steps == 1 else min(5, max_steps - 1)}
  num_steps: 1
  record_shapes: false
run_id: {run_root.name}
output_dir: {run_root}/output
deployment:
  mode: disaggregated
  trainer:
    nnodes: 1
    nproc_per_node: {trainer_nproc}
  disaggregated:
    control_dir: {run_root}/control
    consumer_state_dir: /tmp/{run_root.name}-consumer
    backend: mooncake
    server_urls: [http://127.0.0.1:30000]
    mooncake_metadata_server: http://127.0.0.1:35880/metadata
    mooncake_master_server_addr: 127.0.0.1:35551
    mooncake_protocol: tcp
""".strip()
    config_path = Path("/tmp") / f"{run_root.name}.yaml"
    config_path.write_text(config + "\n")
    return config_path


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_text_command(command: list[str], *, timeout: float = 10) -> dict[str, object]:
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"status": f"{type(exc).__name__}: {exc}", "lines": []}
    return {
        "status": result.returncode,
        "lines": [line for line in result.stdout.splitlines() if line.strip()],
        "stderr": result.stderr.strip(),
    }


def should_commit_while_running(run_root: Path) -> bool:
    return not (run_root / "inference.ready").exists()


def process_thread_environment(rank: int) -> dict[str, str]:
    return {
        "OMP_NUM_THREADS": "16" if rank == 0 else "4",
        "TORCHINDUCTOR_COMPILE_THREADS": str(INDUCTOR_COMPILE_THREADS),
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
    }


def persist_run_snapshot(
    run_root: Path,
    persistent_root: Path,
    *,
    replace: bool,
) -> None:
    collect_rank_logs(run_root)
    if replace and persistent_root.exists():
        shutil.rmtree(persistent_root)
    if run_root.exists():
        shutil.copytree(
            run_root,
            persistent_root,
            dirs_exist_ok=True,
            symlinks=True,
        )
    else:
        persistent_root.mkdir(parents=True, exist_ok=True)


def collect_rank_logs(run_root: Path) -> None:
    if not run_root.exists():
        return
    for rank in (0, 1):
        source = run_root.parent / f"{run_root.name}-rank{rank}.log"
        if source.is_file():
            shutil.copy2(source, run_root / f"rank{rank}.log")
    sampler = run_root.parent / f"{run_root.name}-gpu-sampler.log"
    if sampler.is_file():
        shutil.copy2(sampler, run_root / "gpu-sampler.log")


def launch_gpu_sampler(run_root: Path) -> subprocess.Popen:
    output = run_root.parent / f"{run_root.name}-gpu-sampler.log"
    handle = output.open("w")
    command = (
        "while true; do "
        "date -Ins; "
        "timeout --kill-after=1 5 nvidia-smi "
        f"--query-gpu={','.join(GPU_QUERY_FIELDS)} "
        "--format=csv,noheader,nounits || echo nvidia-smi-status=$?; "
        "sleep 5; "
        "done"
    )
    return subprocess.Popen(
        ["bash", "-lc", command],
        stdout=handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )


def tail_text(path: Path, *, line_count: int = 12) -> list[str]:
    if not path.is_file():
        return []
    return path.read_text(errors="replace").splitlines()[-line_count:]


def compact_runtime_status(status: dict[str, object]) -> dict[str, object]:
    return {
        "event": status["event"],
        "run": status["run"],
        "timestamp": status["timestamp"],
        "elapsed_s": status["elapsed_s"],
        "rank0_status": status["rank0_status"],
        "rank1_status": status["rank1_status"],
        "files": status["files"],
        "checkpoint_states": status["checkpoint_states"],
        "gpu_sampler_tail": status["gpu_sampler_tail"],
    }


def launch_rank(
    *,
    rank: int,
    config: Path,
    run_root: Path,
    server_gpus: str,
    server_tp: int,
    trainer_gpus: str,
    trainer_nproc: int,
    accumulation_steps: int,
    server_backend_args: str,
    attention_backend: str,
    server_mem_fraction: float = 0.60,
) -> subprocess.Popen:
    environment = os.environ.copy()
    environment.update(
        {
            "NODE_RANK": str(rank),
            "NUM_NODES": "2",
            "HEAD_IP": "127.0.0.1",
            "DISAGG_STORE_ID": run_root.name,
            "DISAGG_RUN_ROOT": str(run_root),
            "DISAGG_CONSUMER_STATE_DIR": f"/tmp/{run_root.name}-consumer",
            "CONFIG": str(config),
            "RUN_LABEL": run_root.name,
            "TARGET_MODEL_PATH": TARGET_MODEL,
            "SERVER_GPUS": server_gpus,
            "SERVER_TP": str(server_tp),
            "SERVER_MEM_FRACTION": str(server_mem_fraction),
            "CAPTURE_LAYER_IDS": "6 23 39",
            "TRAINER_GPUS": trainer_gpus,
            "TRAINER_NPROC": str(trainer_nproc),
            "TRAINER_ACCUMULATION_STEPS": str(accumulation_steps),
            "HF_HOME": "/opjax-volume",
            "HF_HUB_CACHE": "/opjax-volume/hub",
            "XDG_CACHE_HOME": "/opjax-volume/xdg",
            "TORCHINDUCTOR_CACHE_DIR": "/opjax-volume/torchinductor",
            "SGLANG_CACHE_DIR": "/opjax-volume/sglang",
            "TVM_FFI_CACHE_DIR": "/opjax-volume/tvm-ffi",
            "TOKENIZERS_PARALLELISM": "false",
            "APPLY_SGLANG_CAPTURE_PATCH": "1",
            "INFERENCE_NODE_IP": "127.0.0.1",
            "TRAINER_NODE_IP": "127.0.0.1",
            "START_TIMEOUT_S": "3600",
            "PEER_TIMEOUT_S": "3600",
            "SERVER_EXTRA_ARGS": (
                "--revision "
                + TARGET_REVISION
                + " --dtype bfloat16 --attention-backend "
                + attention_backend
                + " --context-length 4096 --quantization modelopt_fp4"
                + server_backend_args
                + " --page-size 128 --mamba-radix-cache-strategy extra_buffer"
                + " --max-mamba-cache-size 64 --swa-full-tokens-ratio 0.1"
                + " --mamba-full-memory-ratio 0.1 --disable-cuda-graph"
            ),
        }
    )
    environment.update(process_thread_environment(rank))
    log_path = run_root.parent / f"{run_root.name}-rank{rank}.log"
    log_handle = log_path.open("w")
    return subprocess.Popen(
        ["bash", "/opt/specforge/examples/disagg/run_inkling_dspark_disagg_2node.sh"],
        env=environment,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )


def emit_runtime_status(
    *,
    run_root: Path,
    rank0: subprocess.Popen,
    rank1: subprocess.Popen | None,
    started_at: float,
) -> None:
    output_root = run_root / "output"
    checkpoint_states = (
        sorted(
            str(path.relative_to(run_root))
            for path in output_root.glob("**/training_state*.pt")
        )
        if output_root.exists()
        else []
    )
    tracked_files = [
        run_root / "inference.ready",
        run_root / "producer.log",
        run_root / "consumer.log",
        run_root / "consumer.done",
        run_root / "inference.done",
    ]
    load_average = Path("/proc/loadavg")
    memory_info = Path("/proc/meminfo")
    gpu_sampler = run_root.parent / f"{run_root.name}-gpu-sampler.log"
    status = {
        "event": "opjax_dspark_heartbeat",
        "run": run_root.name,
        "timestamp": datetime.now(UTC).isoformat(),
        "elapsed_s": round(time.monotonic() - started_at, 1),
        "rank0_status": rank0.poll(),
        "rank1_status": None if rank1 is None else rank1.poll(),
        "files": {
            path.name: {"exists": path.exists(), "size": path.stat().st_size}
            if path.exists()
            else {"exists": False, "size": 0}
            for path in tracked_files
        },
        "checkpoint_states": checkpoint_states,
        "gpu_sampler_tail": tail_text(gpu_sampler),
        "load_average": (
            load_average.read_text().strip() if load_average.exists() else None
        ),
        "memory": (
            {
                key: value.strip()
                for line in memory_info.read_text().splitlines()
                if ":" in line
                for key, value in [line.split(":", 1)]
                if key in {"MemAvailable", "MemFree", "MemTotal", "SwapFree"}
            }
            if memory_info.exists()
            else {}
        ),
    }
    encoded = json.dumps(status, sort_keys=True)
    print(json.dumps(compact_runtime_status(status), sort_keys=True), flush=True)
    if run_root.exists():
        with (run_root / "runtime-telemetry.jsonl").open("a") as handle:
            handle.write(encoded + "\n")


def write_runtime_artifacts(
    *,
    run_root: Path,
    config: Path,
    result: dict[str, object],
) -> None:
    run_root.mkdir(parents=True, exist_ok=True)
    (run_root / "config.yaml").write_text(config.read_text())
    pip_freeze = run_text_command(
        [sys.executable, "-m", "pip", "freeze", "--all"], timeout=60
    )
    (run_root / "runtime-pip-freeze.txt").write_text(
        "\n".join(str(line) for line in pip_freeze["lines"]) + "\n"
    )
    nvidia_smi = run_text_command(["nvidia-smi", "-q"], timeout=30)
    (run_root / "runtime-nvidia-smi.txt").write_text(
        "\n".join(str(line) for line in nvidia_smi["lines"]) + "\n"
    )
    source_path = Path(__file__)
    runtime = {
        "generated_at": datetime.now(UTC).isoformat(),
        "result": result,
        "target_model": TARGET_MODEL,
        "target_revision": TARGET_REVISION,
        "specforge_revision": SPECFORGE_REVISION,
        "base_image": BASE_IMAGE,
        "inductor_compile_threads": INDUCTOR_COMPILE_THREADS,
        "python": sys.version,
        "platform": platform.platform(),
        "hostname": socket.gethostname(),
        "launcher_sha256": file_sha256(source_path),
        "config_sha256": file_sha256(run_root / "config.yaml"),
        "pip_freeze_status": pip_freeze["status"],
        "nvidia_smi_status": nvidia_smi["status"],
    }
    (run_root / "runtime.json").write_text(
        json.dumps(runtime, indent=2, sort_keys=True) + "\n"
    )


def write_artifact_manifest(run_root: Path) -> None:
    artifacts_by_path: dict[str, dict[str, object]] = {}
    manifest_path = run_root / "artifact-manifest.json"
    for path in sorted(run_root.rglob("*")):
        relative = str(path.relative_to(run_root))
        if path == manifest_path:
            continue
        if path.is_symlink():
            artifacts_by_path[relative] = {
                "kind": "symlink",
                "target": os.readlink(path),
            }
        elif path.is_file():
            artifacts_by_path[relative] = {
                "kind": "file",
                "size": path.stat().st_size,
                "sha256": file_sha256(path),
            }
    manifest = {
        "generated_at": datetime.now(UTC).isoformat(),
        "run": run_root.name,
        "artifacts": artifacts_by_path,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


@app.function(image=image, timeout=900, secrets=[secret])
def inspect_stack() -> dict[str, str]:
    import sglang
    import specforge
    import torch
    import transformers

    return {
        "hostname": socket.gethostname(),
        "specforge": str(Path(specforge.__file__).resolve()),
        "sglang": str(getattr(sglang, "__version__", "unknown")),
        "torch": str(torch.__version__),
        "transformers": str(transformers.__version__),
        "specforge_revision": SPECFORGE_REVISION,
    }


@app.function(image=image, timeout=900)
def inspect_mount_paths() -> dict[str, object]:
    candidates = [
        "/cache",
        "/mnt/hf-cache",
        "/persistent-cache",
        "/volumes/opjax-hf-cache",
        "/opjax-volume",
    ]
    return {
        "root": sorted(path.name for path in Path("/").iterdir()),
        "candidates": {
            candidate: {
                "exists": Path(candidate).exists(),
                "contents": (
                    sorted(path.name for path in Path(candidate).iterdir())
                    if Path(candidate).is_dir()
                    else []
                ),
            }
            for candidate in candidates
        },
    }


def run_training(
    student: str,
    max_steps: int,
    *,
    server_gpus: str,
    server_tp: int,
    trainer_gpus: str,
    trainer_nproc: int,
    accumulation_steps: int,
    moe_backend: str,
    server_backend_args: str,
    attention_backend: str = "fa4",
    server_mem_fraction: float = 0.60,
) -> dict[str, object]:
    if student not in {"dspark-500m", "dspark-250m"}:
        raise ValueError(f"unknown student: {student}")
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")
    run_name = f"{student}-steps{max_steps}-{int(time.time())}-{uuid.uuid4().hex[:8]}"
    run_root = Path("/tmp/opjax-dspark-runs") / run_name
    persistent_root = Path("/artifacts/runs") / run_name
    run_root.parent.mkdir(parents=True, exist_ok=True)
    persistent_root.parent.mkdir(parents=True, exist_ok=True)
    config = write_config(
        student,
        run_root,
        max_steps,
        trainer_nproc=trainer_nproc,
        accumulation_steps=accumulation_steps,
        moe_backend=moe_backend,
        attention_backend=attention_backend,
        server_mem_fraction=server_mem_fraction,
    )
    rank0 = launch_rank(
        rank=0,
        config=config,
        run_root=run_root,
        server_gpus=server_gpus,
        server_tp=server_tp,
        trainer_gpus=trainer_gpus,
        trainer_nproc=trainer_nproc,
        accumulation_steps=accumulation_steps,
        server_backend_args=server_backend_args,
        attention_backend=attention_backend,
        server_mem_fraction=server_mem_fraction,
    )
    gpu_sampler = launch_gpu_sampler(run_root)
    rank1 = None
    started_at = time.monotonic()
    try:
        time.sleep(5)
        rank1 = launch_rank(
            rank=1,
            config=config,
            run_root=run_root,
            server_gpus=server_gpus,
            server_tp=server_tp,
            trainer_gpus=trainer_gpus,
            trainer_nproc=trainer_nproc,
            accumulation_steps=accumulation_steps,
            server_backend_args=server_backend_args,
            attention_backend=attention_backend,
            server_mem_fraction=server_mem_fraction,
        )
        processes = [rank0, rank1]
        next_status_at = 0.0
        next_commit_at = time.monotonic() + 60
        while any(process.poll() is None for process in processes):
            failed = [
                process
                for process in processes
                if process.poll() is not None and process.returncode != 0
            ]
            if failed:
                break
            if time.monotonic() >= next_status_at:
                try:
                    emit_runtime_status(
                        run_root=run_root,
                        rank0=rank0,
                        rank1=rank1,
                        started_at=started_at,
                    )
                except Exception as exc:
                    print(
                        json.dumps(
                            {
                                "event": "opjax_dspark_heartbeat_error",
                                "error": f"{type(exc).__name__}: {exc}",
                                "run": run_root.name,
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
                next_status_at = time.monotonic() + 30
            if time.monotonic() >= next_commit_at and should_commit_while_running(
                run_root
            ):
                try:
                    persist_run_snapshot(
                        run_root,
                        persistent_root,
                        replace=True,
                    )
                    artifacts.commit()
                    print(
                        json.dumps(
                            {
                                "event": "opjax_dspark_artifacts_committed",
                                "run": run_root.name,
                                "timestamp": datetime.now(UTC).isoformat(),
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
                except Exception as exc:
                    print(
                        json.dumps(
                            {
                                "event": "opjax_dspark_artifact_commit_error",
                                "error": f"{type(exc).__name__}: {exc}",
                                "run": run_root.name,
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
                next_commit_at = time.monotonic() + 60
            time.sleep(1)
    finally:
        for process in [rank0, rank1, gpu_sampler]:
            if process is not None and process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
        for process in [rank0, rank1, gpu_sampler]:
            if process is not None and process.poll() is None:
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
    rank0_status = rank0.returncode
    rank1_status = None if rank1 is None else rank1.returncode
    gpu_name_query = run_text_command(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], timeout=10
    )
    result = {
        "run": run_name,
        "rank0_status": rank0_status,
        "rank1_status": rank1_status,
        "run_root": str(persistent_root),
        "working_root": str(run_root),
        "gpu_names": gpu_name_query["lines"],
        "gpu_name_query_status": gpu_name_query["status"],
        "student": student,
        "max_steps": max_steps,
        "elapsed_s": round(time.monotonic() - started_at, 1),
    }
    (run_root / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    write_runtime_artifacts(run_root=run_root, config=config, result=result)
    collect_rank_logs(run_root)
    write_artifact_manifest(run_root)
    persist_run_snapshot(run_root, persistent_root, replace=True)
    (persistent_root.parent / f"{run_name}-result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    artifacts.commit()
    if rank0_status != 0 or rank1_status != 0:
        raise RuntimeError(json.dumps(result))
    return result


FUNCTION_OPTIONS = {
    "image": image,
    "volumes": {"/opjax-volume": hf_cache, "/artifacts": artifacts},
    "secrets": [secret],
    "cpu": 32.0,
    "memory": 65536,
    "timeout": 24 * 60 * 60,
}


@app.function(gpu="B200:4", **FUNCTION_OPTIONS)
def train(student: str, max_steps: int = 1) -> dict[str, object]:
    return run_training(
        student,
        max_steps,
        server_gpus="0,1",
        server_tp=2,
        trainer_gpus="2,3",
        trainer_nproc=2,
        accumulation_steps=32,
        moe_backend="flashinfer_trtllm_routed",
        server_backend_args=" --moe-runner-backend flashinfer_trtllm_routed",
    )


@app.function(gpu="H200:4", **FUNCTION_OPTIONS)
def train_h200(student: str, max_steps: int = 1) -> dict[str, object]:
    return run_training(
        student,
        max_steps,
        server_gpus="0,1",
        server_tp=2,
        trainer_gpus="2,3",
        trainer_nproc=2,
        accumulation_steps=32,
        moe_backend="marlin",
        server_backend_args=(" --fp4-gemm-backend marlin --moe-runner-backend marlin"),
        server_mem_fraction=0.70,
    )


@app.function(gpu="B200:2", **FUNCTION_OPTIONS)
def train_canary_2b200(student: str, max_steps: int = 1) -> dict[str, object]:
    if max_steps != 1:
        raise ValueError("the 2xB200 fallback is canary-only")
    return run_training(
        student,
        max_steps,
        server_gpus="0",
        server_tp=1,
        trainer_gpus="1",
        trainer_nproc=1,
        accumulation_steps=64,
        moe_backend="flashinfer_trtllm_routed",
        server_backend_args=" --moe-runner-backend flashinfer_trtllm_routed",
    )


@app.function(gpu="H200:2", **FUNCTION_OPTIONS)
def train_canary_2h200(student: str, max_steps: int = 1) -> dict[str, object]:
    if max_steps != 1:
        raise ValueError("the 2xH200 fallback is canary-only")
    return run_training(
        student,
        max_steps,
        server_gpus="0",
        server_tp=1,
        trainer_gpus="1",
        trainer_nproc=1,
        accumulation_steps=64,
        moe_backend="marlin",
        server_backend_args=(" --fp4-gemm-backend marlin --moe-runner-backend marlin"),
    )


@app.function(gpu="H100:2", **FUNCTION_OPTIONS)
def train_canary_2h100(student: str, max_steps: int = 1) -> dict[str, object]:
    if max_steps != 1:
        raise ValueError("the 2xH100 fallback is canary-only")
    return run_training(
        student,
        max_steps,
        server_gpus="0",
        server_tp=1,
        trainer_gpus="1",
        trainer_nproc=1,
        accumulation_steps=64,
        moe_backend="marlin",
        server_backend_args=(" --fp4-gemm-backend marlin --moe-runner-backend marlin"),
    )


@app.function(gpu=["B200:2", "H200:2", "H100:2"], **FUNCTION_OPTIONS)
def train_canary_any_2gpu(student: str, max_steps: int = 1) -> dict[str, object]:
    if max_steps != 1:
        raise ValueError("the adaptive two-GPU fallback is canary-only")
    device_name = subprocess.run(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()[0]
    is_blackwell = "B200" in device_name
    backend = "flashinfer_trtllm_routed" if is_blackwell else "marlin"
    backend_args = (
        " --moe-runner-backend flashinfer_trtllm_routed"
        if is_blackwell
        else " --fp4-gemm-backend marlin --moe-runner-backend marlin"
    )
    return run_training(
        student,
        max_steps,
        server_gpus="0",
        server_tp=1,
        trainer_gpus="1",
        trainer_nproc=1,
        accumulation_steps=64,
        moe_backend=backend,
        server_backend_args=backend_args,
    )


@app.function(gpu=["B200", "H200", "H100", "A100-80GB"], **FUNCTION_OPTIONS)
def train_canary_colocated(student: str, max_steps: int = 1) -> dict[str, object]:
    if max_steps != 1:
        raise ValueError("the colocated one-GPU fallback is canary-only")
    device_name = subprocess.run(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()[0]
    is_blackwell = "B200" in device_name
    attention_backend = "flashinfer" if "A100" in device_name else "fa4"
    backend = "flashinfer_trtllm_routed" if is_blackwell else "marlin"
    backend_args = (
        " --moe-runner-backend flashinfer_trtllm_routed"
        if is_blackwell
        else " --fp4-gemm-backend marlin --moe-runner-backend marlin"
    )
    return run_training(
        student,
        max_steps,
        server_gpus="0",
        server_tp=1,
        trainer_gpus="0",
        trainer_nproc=1,
        accumulation_steps=64,
        moe_backend=backend,
        server_backend_args=backend_args,
        attention_backend=attention_backend,
    )


@app.function(gpu="L40S", **FUNCTION_OPTIONS)
def train_canary_l40s(student: str, max_steps: int = 1) -> dict[str, object]:
    if max_steps != 1:
        raise ValueError("the L40S fallback is canary-only")
    return run_training(
        student,
        max_steps,
        server_gpus="0",
        server_tp=1,
        trainer_gpus="0",
        trainer_nproc=1,
        accumulation_steps=64,
        moe_backend="marlin",
        server_backend_args=(" --fp4-gemm-backend marlin --moe-runner-backend marlin"),
        attention_backend="flashinfer",
    )
