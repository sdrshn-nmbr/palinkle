from __future__ import annotations

import subprocess

import modal


APP_NAME = "opjax-inkling-small-tp4-specbench"
MODEL_ID = "thinkingmachines/Inkling-Small-NVFP4"
MODEL_REVISION = "b6a99534467840620d411e4cd4ad5819b2610d9c"
DRAFT_ID = "RadixArk/Inkling-Small-DSpark"
DRAFT_REVISION = "736501c3901cfc6bbb53ba382781eb0e5d9ad66a"
BF16_MODEL_ID = "thinkingmachines/Inkling-Small"
BF16_MODEL_REVISION = "8cc5877b44d343f88b92086aa1fb72897950f06a"
IMAGE = "lmsysorg/sglang@sha256:b90c0d760a65bc4dbbe4520bea966c437cc40391dcb7cca2a74922985dc1abeb"
PORT = 8000

app = modal.App(APP_NAME)
cache = modal.Volume.from_name(
    "opjax-hf-cache-v2",
    environment_name="main",
    create_if_missing=True,
    version=2,
)
secret = modal.Secret.from_name("opjax-secrets", environment_name="main")
image = modal.Image.from_registry(IMAGE).env(
    {
        "HF_HOME": "/mnt/hf-cache",
        "HF_HUB_CACHE": "/mnt/hf-cache/hub",
        "HF_XET_HIGH_PERFORMANCE": "1",
        "SGLANG_ENABLE_UNIFIED_RADIX_TREE": "1",
    }
)


def command(strategy: str) -> list[str]:
    args = [
        "sglang",
        "serve",
        "--model-path",
        MODEL_ID,
        "--revision",
        MODEL_REVISION,
        "--served-model-name",
        MODEL_ID,
        "--host",
        "0.0.0.0",
        "--port",
        str(PORT),
        "--tp",
        "4",
        "--quantization",
        "modelopt_fp4",
        "--attention-backend",
        "fa4",
        "--page-size",
        "128",
        "--fp4-gemm-backend",
        "marlin",
        "--moe-runner-backend",
        "marlin",
        "--enable-torch-symm-mem",
        "--mamba-radix-cache-strategy",
        "extra_buffer",
        "--swa-full-tokens-ratio",
        "0.1",
        "--mamba-full-memory-ratio",
        "0.1",
        "--context-length",
        "32768",
        "--max-running-requests",
        "16",
        "--cuda-graph-max-bs-decode",
        "16",
        "--cuda-graph-bs-decode",
        "1",
        "4",
        "16",
        "--reasoning-parser",
        "inkling",
        "--tool-call-parser",
        "inkling",
        "--enable-metrics",
        "--enable-mfu-metrics",
        "--decode-log-interval",
        "20",
        "--random-seed",
        "0",
    ]
    if strategy == "balanced":
        return args + ["--mem-fraction-static", "0.85"]
    if strategy == "mtp":
        return args + [
            "--mem-fraction-static",
            "0.60",
            "--speculative-algorithm",
            "EAGLE",
            "--speculative-num-steps",
            "8",
            "--speculative-eagle-topk",
            "1",
            "--speculative-num-draft-tokens",
            "9",
            "--enable-multi-layer-eagle",
            "--speculative-use-rejection-sampling",
        ]
    if strategy == "dspark":
        return args + [
            "--mem-fraction-static",
            "0.60",
            "--skip-server-warmup",
            "--speculative-algorithm",
            "DSPARK",
            "--speculative-draft-model-path",
            DRAFT_ID,
            "--speculative-draft-model-revision",
            DRAFT_REVISION,
            "--speculative-draft-model-quantization",
            "unquant",
            "--speculative-dspark-block-size",
            "7",
            "--chunked-prefill-size",
            "8192",
            "--cuda-graph-bs-prefill",
            "128",
            "512",
            "2048",
            "--disable-flashinfer-autotune",
        ]
    raise ValueError(f"unknown strategy: {strategy}")


def launch(strategy: str) -> None:
    subprocess.Popen(command(strategy))


def launch_bf16() -> None:
    subprocess.Popen(
        [
            "sglang",
            "serve",
            "--model-path",
            BF16_MODEL_ID,
            "--revision",
            BF16_MODEL_REVISION,
            "--served-model-name",
            BF16_MODEL_ID,
            "--host",
            "0.0.0.0",
            "--port",
            str(PORT),
            "--tp",
            "8",
            "--dtype",
            "bfloat16",
            "--mem-fraction-static",
            "0.85",
            "--context-length",
            "32768",
            "--max-running-requests",
            "16",
            "--cuda-graph-max-bs-decode",
            "16",
            "--cuda-graph-bs-decode",
            "1",
            "4",
            "16",
            "--disable-prefill-cuda-graph",
            "--trust-remote-code",
            "--mamba-radix-cache-strategy",
            "extra_buffer",
            "--swa-full-tokens-ratio",
            "0.1",
            "--mamba-full-memory-ratio",
            "0.1",
            "--reasoning-parser",
            "inkling",
            "--tool-call-parser",
            "inkling",
            "--enable-metrics",
            "--enable-mfu-metrics",
            "--decode-log-interval",
            "20",
            "--random-seed",
            "0",
        ]
    )


COMMON = {
    "image": image,
    "gpu": "H200:4",
    "volumes": {"/mnt/hf-cache": cache},
    "secrets": [secret],
    "timeout": 3600,
    "startup_timeout": 3600,
    "scaledown_window": 120,
    "max_containers": 1,
}


@app.function(
    image=image,
    gpu="H200:8",
    volumes={"/mnt/hf-cache": cache},
    secrets=[secret],
    timeout=3600,
    startup_timeout=3600,
    scaledown_window=120,
    max_containers=1,
)
@modal.concurrent(max_inputs=68)
@modal.web_server(PORT, startup_timeout=3600, requires_proxy_auth=True)
def bf16() -> None:
    launch_bf16()


@app.function(**COMMON)
@modal.concurrent(max_inputs=68)
@modal.web_server(PORT, startup_timeout=3600, requires_proxy_auth=True)
def balanced() -> None:
    launch("balanced")


@app.function(**COMMON)
@modal.concurrent(max_inputs=68)
@modal.web_server(PORT, startup_timeout=3600, requires_proxy_auth=True)
def mtp() -> None:
    launch("mtp")


@app.function(**COMMON)
@modal.concurrent(max_inputs=68)
@modal.web_server(PORT, startup_timeout=3600, requires_proxy_auth=True)
def dspark() -> None:
    launch("dspark")
