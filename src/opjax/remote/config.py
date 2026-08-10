"""Shared Modal configuration for reproducible remote runs."""

from __future__ import annotations

import os
import subprocess


MODAL_APP_NAME = "opjax"
MODAL_PROFILE = "conway"
MODAL_ENVIRONMENT = "main"
MODAL_SECRET_NAME = "opjax-secrets"

PYTHON_VERSION = "3.12"
GPU_TYPE = "H200"

# Modal Volumes v2 are the default. Set OPJAX_MODAL_VOLUME_VERSION=1 to use the
# v1 fallback names if v2 blocks us during an experiment.
MODAL_VOLUME_VERSION = int(os.environ.get("OPJAX_MODAL_VOLUME_VERSION", "2"))
if MODAL_VOLUME_VERSION not in {1, 2}:
    raise ValueError("OPJAX_MODAL_VOLUME_VERSION must be 1 or 2")

VOLUME_SUFFIX = f"v{MODAL_VOLUME_VERSION}"
HF_CACHE_VOLUME_NAME = f"opjax-hf-cache-{VOLUME_SUFFIX}"
DATA_VOLUME_NAME = f"opjax-data-{VOLUME_SUFFIX}"
CHECKPOINT_VOLUME_NAME = f"opjax-checkpoints-{VOLUME_SUFFIX}"

HF_CACHE_DIR = "/mnt/hf-cache"
DATA_DIR = "/mnt/data"
CHECKPOINT_DIR = "/mnt/checkpoints"

EXPECTED_SECRET_KEYS = ("HF_TOKEN", "ANTHROPIC_API_KEY", "WANDB_API_KEY")

MODAL_PROXY_TOKEN_ENV = "MODAL_PROXY_TOKEN"
MODAL_PROXY_KEYCHAIN_ACCOUNT = "opjax"
MODAL_PROXY_KEYCHAIN_SERVICE = "com.opjax.modal.proxy.main"


def _modal_proxy_token() -> str:
    token = os.environ.get(MODAL_PROXY_TOKEN_ENV, "").strip()
    if not token:
        result = subprocess.run(
            [
                "/usr/bin/security",
                "find-generic-password",
                "-a",
                MODAL_PROXY_KEYCHAIN_ACCOUNT,
                "-s",
                MODAL_PROXY_KEYCHAIN_SERVICE,
                "-w",
            ],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            token = result.stdout.strip()
    token_id, separator, token_secret = token.partition(".")
    if (
        separator != "."
        or not token_id.startswith("wk-")
        or not token_secret.startswith("ws-")
    ):
        raise RuntimeError("MODAL_PROXY_TOKEN_MISSING")
    return token


def modal_proxy_headers() -> dict[str, str]:
    token_id, token_secret = _modal_proxy_token().split(".", maxsplit=1)
    return {"Modal-Key": token_id, "Modal-Secret": token_secret}

REMOTE_IMAGE_PACKAGES = (
    "anthropic==0.97.0",
    "chex==0.1.91",
    "dialog @ git+https://github.com/google-deepmind/dialog.git",
    "gemma @ git+https://github.com/google-deepmind/gemma.git",
    "google-tunix @ git+https://github.com/sdrshn-nmbr/tunix.git@opjax/gemma4-vision-port",
    "hf-transfer==0.1.9",
    "huggingface-hub==0.36.2",
    "jax[cuda12]==0.10.0",
    "jaxtyping==0.3.9",
    "orbax-checkpoint==0.11.39",
    "pillow==12.2.0",
    "qwix>=0.1.6",
    "wandb==0.26.1",
)

GEMMA4_GCS_BUCKET = "gs://gemma-data/checkpoints"
GEMMA4_26B_A4B_IT = f"{GEMMA4_GCS_BUCKET}/gemma4-26b-a4b-it"

REMOTE_ENV = {
    "HF_HOME": HF_CACHE_DIR,
    "HF_HUB_CACHE": f"{HF_CACHE_DIR}/hub",
    "HF_HUB_ENABLE_HF_TRANSFER": "1",
    "OPJAX_DATA_DIR": DATA_DIR,
    "OPJAX_CHECKPOINT_DIR": CHECKPOINT_DIR,
    "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
}
