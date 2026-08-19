"""Modal endpoints for matched Laguna plain, DFlash, and DSpark inference."""

from __future__ import annotations

import os
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
TRAINING_ROOT = "/mnt/training/experiments/serving-native-v2"
SERVING_GPU = os.environ.get("OPJAX_LAGUNA_SERVING_GPU", "H200")
SERVING_ATTEMPT_ID = os.environ.get(
    "OPJAX_LAGUNA_SERVING_ATTEMPT_ID",
    "20260818-serving-native-v2-h200-v1",
)
if not SERVING_ATTEMPT_ID.replace("-", "").isalnum():
    raise ValueError(f"LAGUNA_SERVING_ATTEMPT_ID_INVALID:{SERVING_ATTEMPT_ID}")

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
    .uv_pip_install("huggingface-hub==1.4.1")
    .env(
        {
            **REMOTE_ENV,
            "OPJAX_SPEC_ARTIFACT_ROOT": ARTIFACT_ROOT,
            "OPJAX_SPEC_ARTIFACT_VOLUME": "opjax-laguna-speculative-artifacts-v1",
            "OPJAX_SPEC_MODAL_ENVIRONMENT": MODAL_ENVIRONMENT,
            "OPJAX_SPEC_ATTEMPT_ID": SERVING_ATTEMPT_ID,
            "OPJAX_SPEC_DECLARED_GPU": SERVING_GPU,
        }
    )
    .add_local_python_source("opjax")
)

FUNCTION_OPTIONS = {
    "image": image,
    "gpu": SERVING_GPU,
    "volumes": {
        HF_CACHE_DIR: cache,
        ARTIFACT_ROOT: artifacts,
        TRAINING_ROOT: training,
    },
    "secrets": [secret],
    "timeout": 3600,
    "startup_timeout": 3600,
    "scaledown_window": 60,
    "max_containers": 1,
}


def _launch(
    arm: str,
    *,
    proposal_tokens: int | None = None,
    adaptive_verification: bool | None = None,
    draft_model: str | None = None,
) -> None:
    checkpoint = "trained" if draft_model is not None else "released"
    schedule = "adaptive" if adaptive_verification else "fixed"
    suffix = (
        f"{checkpoint}-{arm}"
        if proposal_tokens is None
        else f"{checkpoint}-{arm}-{schedule}-{proposal_tokens}"
    )
    os.environ["OPJAX_SPEC_RUN_ID"] = f"{SERVING_ATTEMPT_ID}-{suffix}"
    subprocess.Popen(
        server_command(
            arm,
            port=PORT,
            proposal_tokens=proposal_tokens,
            adaptive_verification=adaptive_verification,
            draft_model=draft_model,
        )
    )


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


@app.function(**FUNCTION_OPTIONS)
@modal.concurrent(max_inputs=32)
@modal.web_server(PORT, startup_timeout=3600, requires_proxy_auth=True)
def dspark4() -> None:
    _launch(DSPARK, proposal_tokens=4, adaptive_verification=False)


@app.function(**FUNCTION_OPTIONS)
@modal.concurrent(max_inputs=32)
@modal.web_server(PORT, startup_timeout=3600, requires_proxy_auth=True)
def dspark8() -> None:
    _launch(DSPARK, proposal_tokens=8, adaptive_verification=False)


@app.function(**FUNCTION_OPTIONS)
@modal.concurrent(max_inputs=32)
@modal.web_server(PORT, startup_timeout=3600, requires_proxy_auth=True)
def dspark12() -> None:
    _launch(DSPARK, proposal_tokens=12, adaptive_verification=False)


@app.function(**FUNCTION_OPTIONS)
@modal.concurrent(max_inputs=32)
@modal.web_server(PORT, startup_timeout=3600, requires_proxy_auth=True)
def dspark15() -> None:
    _launch(DSPARK, proposal_tokens=15, adaptive_verification=False)


def _trained_dflash(depth: int) -> None:
    _launch(
        DFLASH,
        proposal_tokens=depth,
        draft_model=f"{TRAINING_ROOT}/selected/dflash",
    )


def _trained_dspark(depth: int, *, adaptive: bool = False) -> None:
    _launch(
        DSPARK,
        proposal_tokens=depth,
        adaptive_verification=adaptive,
        draft_model=f"{TRAINING_ROOT}/selected/dspark",
    )


@app.function(**FUNCTION_OPTIONS)
@modal.concurrent(max_inputs=32)
@modal.web_server(PORT, startup_timeout=3600, requires_proxy_auth=True)
def trained_dflash4() -> None:
    _trained_dflash(4)


@app.function(**FUNCTION_OPTIONS)
@modal.concurrent(max_inputs=32)
@modal.web_server(PORT, startup_timeout=3600, requires_proxy_auth=True)
def trained_dflash8() -> None:
    _trained_dflash(8)


@app.function(**FUNCTION_OPTIONS)
@modal.concurrent(max_inputs=32)
@modal.web_server(PORT, startup_timeout=3600, requires_proxy_auth=True)
def trained_dflash12() -> None:
    _trained_dflash(12)


@app.function(**FUNCTION_OPTIONS)
@modal.concurrent(max_inputs=32)
@modal.web_server(PORT, startup_timeout=3600, requires_proxy_auth=True)
def trained_dflash15() -> None:
    _trained_dflash(15)


@app.function(**FUNCTION_OPTIONS)
@modal.concurrent(max_inputs=32)
@modal.web_server(PORT, startup_timeout=3600, requires_proxy_auth=True)
def trained_dspark4() -> None:
    _trained_dspark(4)


@app.function(**FUNCTION_OPTIONS)
@modal.concurrent(max_inputs=32)
@modal.web_server(PORT, startup_timeout=3600, requires_proxy_auth=True)
def trained_dspark8() -> None:
    _trained_dspark(8)


@app.function(**FUNCTION_OPTIONS)
@modal.concurrent(max_inputs=32)
@modal.web_server(PORT, startup_timeout=3600, requires_proxy_auth=True)
def trained_dspark12() -> None:
    _trained_dspark(12)


@app.function(**FUNCTION_OPTIONS)
@modal.concurrent(max_inputs=32)
@modal.web_server(PORT, startup_timeout=3600, requires_proxy_auth=True)
def trained_dspark15() -> None:
    _trained_dspark(15)


@app.function(**FUNCTION_OPTIONS)
@modal.concurrent(max_inputs=32)
@modal.web_server(PORT, startup_timeout=3600, requires_proxy_auth=True)
def trained_dspark_adaptive() -> None:
    _trained_dspark(15, adaptive=True)
