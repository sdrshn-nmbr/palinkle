"""Modal-hosted OpenAI-compatible SGLang server for Inkling Small."""

from __future__ import annotations

import subprocess

import modal

from opjax.pallas.model_registry import INKLING_HF_REVISION, INKLING_MODEL_ID
from opjax.remote.config import (
    HF_CACHE_DIR,
    HF_CACHE_VOLUME_NAME,
    MODAL_ENVIRONMENT,
    MODAL_SECRET_NAME,
    MODAL_VOLUME_VERSION,
    REMOTE_ENV,
)

MODEL_ID = INKLING_MODEL_ID
MODEL_REVISION = INKLING_HF_REVISION
SGLANG_IMAGE = (
    "lmsysorg/sglang@sha256:b90c0d760a65bc4dbbe4520bea966c437cc40391dcb7cca2a74922985dc1abeb"
)
SGLANG_REVISION = "b7252cc6b0c78b25ecea7ee5efa91a6ae37d0f19"
PRECISION = "bfloat16"
PORT = 8000
ENDPOINT_URL = "https://conway--opjax-inkling-small-openai-serve.modal.run"

app = modal.App("opjax-inkling-small-openai")
hf_cache_volume = modal.Volume.from_name(
    HF_CACHE_VOLUME_NAME,
    environment_name=MODAL_ENVIRONMENT,
    create_if_missing=True,
    version=MODAL_VOLUME_VERSION,
)
opjax_secret = modal.Secret.from_name(
    MODAL_SECRET_NAME,
    environment_name=MODAL_ENVIRONMENT,
)
image = modal.Image.from_registry(SGLANG_IMAGE).env(
    {**REMOTE_ENV, "SGLANG_ENABLE_UNIFIED_RADIX_TREE": "1"}
)


@app.function(
    image=image,
    gpu="H200:8",
    volumes={HF_CACHE_DIR: hf_cache_volume},
    secrets=[opjax_secret],
    timeout=3600,
    startup_timeout=3600,
    scaledown_window=300,
    max_containers=1,
)
@modal.concurrent(max_inputs=64)
@modal.web_server(
    PORT,
    startup_timeout=3600,
    requires_proxy_auth=True,
)
def serve() -> None:
    command = [
        "python",
        "-m",
        "sglang.launch_server",
        "--model-path",
        MODEL_ID,
        "--revision",
        MODEL_REVISION,
        "--host",
        "0.0.0.0",
        "--port",
        str(PORT),
        "--tp",
        "8",
        "--mem-fraction-static",
        "0.85",
        "--context-length",
        "32768",
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
        "--dtype",
        PRECISION,
        "--random-seed",
        "0",
    ]
    subprocess.Popen(command)
