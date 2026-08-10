# pyright: reportMissingImports=false
"""Modal-hosted SGLang engine for Laguna XS 2.1."""

from __future__ import annotations

import subprocess
import time
from typing import Any

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
SGLANG_IMAGE = "lmsysorg/sglang:v0.5.16"
SGLANG_REVISION = "v0.5.16"
PRECISION = "bfloat16"

app = modal.App("opjax-laguna-sglang")
hf_cache_volume = modal.Volume.from_name(
    HF_CACHE_VOLUME_NAME,
    environment_name=MODAL_ENVIRONMENT,
    create_if_missing=True,
    version=MODAL_VOLUME_VERSION,
)
opjax_secret = modal.Secret.from_name(MODAL_SECRET_NAME, environment_name=MODAL_ENVIRONMENT)
image = (
    modal.Image.from_registry(SGLANG_IMAGE)
    .env(REMOTE_ENV)
    .add_local_python_source("opjax")
)


@app.cls(
    image=image,
    gpu="H200",
    volumes={HF_CACHE_DIR: hf_cache_volume},
    secrets=[opjax_secret],
    timeout=3600,
    startup_timeout=3600,
    scaledown_window=300,
    max_containers=1,
)
@modal.concurrent(max_inputs=4, target_inputs=2)
class LagunaEngine:
    @modal.enter()
    def load(self) -> None:
        import sglang as sgl
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_ID,
            revision=MODEL_REVISION,
            trust_remote_code=True,
        )
        self.engine = sgl.Engine(
            model_path=MODEL_ID,
            revision=MODEL_REVISION,
            tp_size=1,
            mem_fraction_static=0.72,
            context_length=32768,
            trust_remote_code=True,
            reasoning_parser="poolside_v1",
            dtype=PRECISION,
            random_seed=0,
        )

    @modal.method()
    def generate(
        self,
        messages: list[dict[str, str]],
        sampling: dict[str, Any],
    ) -> dict[str, Any]:
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        started = time.perf_counter()
        response = self.engine.generate(prompt=prompt, sampling_params=sampling)
        latency = time.perf_counter() - started
        meta = response.get("meta_info", {})
        finish = meta.get("finish_reason") or {}
        return {
            "text": response["text"],
            "prompt_tokens": meta.get("prompt_tokens"),
            "completion_tokens": meta.get("completion_tokens"),
            "stop_reason": finish.get("type") if isinstance(finish, dict) else str(finish),
            "latency_seconds": latency,
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "runtime_revision": SGLANG_REVISION,
            "precision": PRECISION,
        }

    @modal.method()
    def smoke(self) -> dict[str, Any]:
        process = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.used", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
        )
        return {
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "runtime_revision": SGLANG_REVISION,
            "precision": PRECISION,
            "nvidia_smi": process.stdout.strip(),
        }
