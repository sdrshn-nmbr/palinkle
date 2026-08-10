"""Modal-hosted OpenAI-compatible SGLang server for Laguna XS 2.1."""

from __future__ import annotations

import subprocess

import modal

from opjax.remote.config import (
    HF_CACHE_DIR,
    HF_CACHE_VOLUME_NAME,
    MODAL_ENVIRONMENT,
    MODAL_SECRET_NAME,
    MODAL_VOLUME_VERSION,
    REMOTE_ENV,
)

MODEL_ID = "poolside/Laguna-XS-2.1"
MODEL_REVISION = "e9df9a59996d790b94b70f3fef343fe1d9e34bdf"
SGLANG_IMAGE = (
    "lmsysorg/sglang@sha256:984699c298a95b73c469b2191403ddc85fd780506e13c39c4afff3845e27bc6c"
)
SGLANG_REVISION = "v0.5.16"
PRECISION = "bfloat16"
PORT = 8000
ENDPOINT_URL = "https://conway--opjax-laguna-openai-serve.modal.run"

app = modal.App("opjax-laguna-openai")
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
image = modal.Image.from_registry(SGLANG_IMAGE).env(REMOTE_ENV)


@app.function(
    image=image,
    gpu="H200",
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
        "1",
        "--mem-fraction-static",
        "0.72",
        "--context-length",
        "32768",
        "--trust-remote-code",
        "--reasoning-parser",
        "poolside_v1",
        "--tool-call-parser",
        "poolside_v1",
        "--dtype",
        PRECISION,
        "--random-seed",
        "0",
    ]
    subprocess.Popen(command)
