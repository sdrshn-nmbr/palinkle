"""Modal endpoints for matched Laguna plain, DFlash, and DSpark inference."""

from __future__ import annotations

import subprocess

import modal

from opjax.pallas.laguna_speculative import (
    DFLASH,
    DSPARK,
    PLAIN,
    VLLM_IMAGE,
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

APP_NAME = "opjax-laguna-speculative-v1"
PORT = 8000
ARTIFACT_ROOT = "/mnt/spec-artifacts"

app = modal.App(APP_NAME)
cache = modal.Volume.from_name(
    HF_CACHE_VOLUME_NAME,
    environment_name=MODAL_ENVIRONMENT,
    create_if_missing=True,
    version=MODAL_VOLUME_VERSION,
)
artifacts = modal.Volume.from_name(
    "opjax-laguna-speculative-artifacts-v1",
    environment_name=MODAL_ENVIRONMENT,
    create_if_missing=True,
    version=1,
)
secret = modal.Secret.from_name(MODAL_SECRET_NAME, environment_name=MODAL_ENVIRONMENT)
image = (
    modal.Image.from_registry(VLLM_IMAGE, add_python="3.12")
    .entrypoint([])
    .uv_pip_install("huggingface-hub==1.4.1")
    .env({**REMOTE_ENV, "OPJAX_SPEC_ARTIFACT_ROOT": ARTIFACT_ROOT})
    .add_local_python_source("opjax")
)

FUNCTION_OPTIONS = {
    "image": image,
    "gpu": "H200",
    "volumes": {HF_CACHE_DIR: cache, ARTIFACT_ROOT: artifacts},
    "secrets": [secret],
    "timeout": 3600,
    "startup_timeout": 3600,
    "scaledown_window": 1800,
    "max_containers": 1,
}


def _launch(arm: str) -> None:
    subprocess.Popen(server_command(arm, port=PORT))


@app.function(**FUNCTION_OPTIONS)
@modal.concurrent(max_inputs=32)
@modal.web_server(PORT, startup_timeout=3600, requires_proxy_auth=True)
def plain() -> None:
    _launch(PLAIN)


@app.function(**FUNCTION_OPTIONS)
@modal.concurrent(max_inputs=32)
@modal.web_server(PORT, startup_timeout=3600, requires_proxy_auth=True)
def dflash() -> None:
    _launch(DFLASH)


@app.function(**FUNCTION_OPTIONS)
@modal.concurrent(max_inputs=32)
@modal.web_server(PORT, startup_timeout=3600, requires_proxy_auth=True)
def dspark() -> None:
    _launch(DSPARK)
