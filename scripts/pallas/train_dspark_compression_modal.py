from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import time
import uuid
from pathlib import Path

import modal


APP_NAME = "opjax-dspark-compression-training"
BASE_IMAGE = "lmsysorg/sglang@sha256:b90c0d760a65bc4dbbe4520bea966c437cc40391dcb7cca2a74922985dc1abeb"
SPECFORGE_REVISION = "e6440f09a8574b35f894608559fd3d165971e488"
TARGET_MODEL = "thinkingmachines/Inkling-Small-NVFP4"
TARGET_REVISION = "b6a99534467840620d411e4cd4ad5819b2610d9c"

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
            "OMP_NUM_THREADS": "4",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "APPLY_SGLANG_CAPTURE_PATCH": "1",
            "INFERENCE_NODE_IP": "127.0.0.1",
            "TRAINER_NODE_IP": "127.0.0.1",
            "START_TIMEOUT_S": "3600",
            "PEER_TIMEOUT_S": "3600",
            "SERVER_EXTRA_ARGS": (
                "--revision " + TARGET_REVISION
                + " --dtype bfloat16 --attention-backend " + attention_backend
                + " --context-length 4096 --quantization modelopt_fp4"
                + server_backend_args
                + " --page-size 128 --mamba-radix-cache-strategy extra_buffer"
                + " --max-mamba-cache-size 64 --swa-full-tokens-ratio 0.1"
                + " --mamba-full-memory-ratio 0.1 --disable-cuda-graph"
            ),
        }
    )
    log_path = run_root.parent / f"{run_root.name}-rank{rank}.log"
    log_handle = log_path.open("w")
    return subprocess.Popen(
        ["bash", "/opt/specforge/examples/disagg/run_inkling_dspark_disagg_2node.sh"],
        env=environment,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )


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
    run_name = (
        f"{student}-steps{max_steps}-{int(time.time())}-{uuid.uuid4().hex[:8]}"
    )
    run_root = Path("/artifacts/runs") / run_name
    run_root.parent.mkdir(parents=True, exist_ok=True)
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
    rank1 = None
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
        while any(process.poll() is None for process in processes):
            failed = [
                process
                for process in processes
                if process.poll() is not None and process.returncode != 0
            ]
            if failed:
                break
            time.sleep(1)
    finally:
        for process in [rank0, rank1]:
            if process is not None and process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
        for process in [rank0, rank1]:
            if process is not None and process.poll() is None:
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
    rank0_status = rank0.returncode
    rank1_status = None if rank1 is None else rank1.returncode
    artifacts.commit()
    gpu_names = subprocess.run(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    result = {
        "run": run_name,
        "rank0_status": rank0_status,
        "rank1_status": rank1_status,
        "run_root": str(run_root),
        "gpu_names": gpu_names,
    }
    (run_root.parent / f"{run_name}-result.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    artifacts.commit()
    if rank0_status != 0 or rank1_status != 0:
        raise RuntimeError(json.dumps(result))
    return result


FUNCTION_OPTIONS = {
    "image": image,
    "volumes": {"/opjax-volume": hf_cache, "/artifacts": artifacts},
    "secrets": [secret],
    "cpu": 16.0,
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
        server_backend_args=(
            " --fp4-gemm-backend marlin --moe-runner-backend marlin"
        ),
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
        server_backend_args=(
            " --fp4-gemm-backend marlin --moe-runner-backend marlin"
        ),
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
        server_backend_args=(
            " --fp4-gemm-backend marlin --moe-runner-backend marlin"
        ),
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
        server_backend_args=(
            " --fp4-gemm-backend marlin --moe-runner-backend marlin"
        ),
        attention_backend="flashinfer",
    )
