"""Modal hardware lanes for Laguna DSpark differential conformance."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import uuid

import modal

from opjax.remote.config import (
    HF_CACHE_DIR,
    HF_CACHE_VOLUME_NAME,
    MODAL_ENVIRONMENT,
    MODAL_SECRET_NAME,
    MODAL_VOLUME_VERSION,
    REMOTE_ENV,
)


APP_NAME = "opjax-laguna-dspark-conformance-v1"
ARTIFACT_ROOT = "/mnt/conformance"
DEEPSPEC_REVISION = "787db11ea347ac3944233e5aa9c7f1bd8a9b5ced"
DEFAULT_PROMPT = "Use the bash tool to print the current working directory."

app = modal.App(APP_NAME)
cache = modal.Volume.from_name(
    HF_CACHE_VOLUME_NAME,
    environment_name=MODAL_ENVIRONMENT,
    create_if_missing=True,
    version=MODAL_VOLUME_VERSION,
)
artifacts = modal.Volume.from_name(
    "opjax-laguna-dspark-conformance-v1",
    environment_name=MODAL_ENVIRONMENT,
    create_if_missing=True,
    version=MODAL_VOLUME_VERSION,
)
secret = modal.Secret.from_name(MODAL_SECRET_NAME, environment_name=MODAL_ENVIRONMENT)

vllm_image = (
    modal.Image.from_registry(
        "vllm/vllm-openai:nightly@sha256:df1979d8cfbc7e09da32ee568e2c189a76378db7894c5ae55d8eeb99e2be8f1b",
        add_python="3.12",
    )
    .entrypoint([])
    .uv_pip_install("huggingface-hub==1.4.1", "numpy==2.4.4")
    .env(
        {
            **REMOTE_ENV,
            "OPJAX_SPEC_ARTIFACT_ROOT": ARTIFACT_ROOT,
            "OPJAX_SPEC_ARTIFACT_VOLUME": "opjax-laguna-dspark-conformance-v1",
            "OPJAX_SPEC_MODAL_ENVIRONMENT": MODAL_ENVIRONMENT,
        }
    )
    .add_local_python_source("opjax")
)

deepspec_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .uv_pip_install(
        "torch==2.9.1",
        "transformers==5.10.2",
        "numpy==2.4.4",
        "PyYAML==6.0.3",
        "tqdm==4.67.3",
        "triton==3.5.1",
        "typing_extensions==4.15.0",
        "sentencepiece==0.2.1",
        "safetensors==0.7.0",
        "prettytable==3.17.0",
        "compressed-tensors==0.15.0.1",
        "psutil==7.2.2",
        "accelerate==1.14.0",
        "huggingface-hub==1.5.0",
    )
    .run_commands(
        "git clone https://github.com/RespectMathias/DeepSpec.git /opt/deepspec",
        f"git -C /opt/deepspec checkout {DEEPSPEC_REVISION}",
    )
    .env({**REMOTE_ENV, "PYTHONPATH": "/opt/deepspec"})
    .add_local_python_source("opjax")
)

FUNCTION_OPTIONS = {
    "gpu": "H200",
    "volumes": {HF_CACHE_DIR: cache, ARTIFACT_ROOT: artifacts},
    "secrets": [secret],
    "timeout": 3600,
}


def _start_telemetry(path: Path) -> tuple[subprocess.Popen[bytes], object]:
    output = path.open("wb")
    process = subprocess.Popen(
        [
            "nvidia-smi",
            "--query-gpu=timestamp,index,name,memory.used,memory.total,utilization.gpu,power.draw",
            "--format=csv",
            "--loop=1",
        ],
        stdout=output,
        stderr=subprocess.STDOUT,
    )
    return process, output


@app.function(image=deepspec_image, **FUNCTION_OPTIONS)
def capture_deepspec(run_id: str, prompt: str = DEFAULT_PROMPT) -> dict[str, object]:
    root = Path(ARTIFACT_ROOT) / run_id / "deepspec"
    root.parent.mkdir(parents=True, exist_ok=True)
    telemetry, telemetry_output = _start_telemetry(root.parent / "deepspec-gpu.csv")
    log_path = root.parent / "deepspec.log"
    try:
        with log_path.open("wb") as log:
            subprocess.run(
                [
                    "python",
                    "-m",
                    "opjax.remote.laguna_deepspec_conformance",
                    "--output-root",
                    str(root),
                    "--prompt",
                    prompt,
                ],
                check=True,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
    finally:
        telemetry.terminate()
        telemetry.wait(timeout=30)
        telemetry_output.close()
    artifacts.commit()
    return json.loads((root / "manifest.json").read_text(encoding="utf-8"))


@app.function(image=vllm_image, **FUNCTION_OPTIONS)
def capture_vllm(run_id: str, prompt: str = DEFAULT_PROMPT) -> dict[str, object]:
    from opjax.remote.laguna_vllm_conformance import run_capture

    root = Path(ARTIFACT_ROOT) / run_id / "vllm"
    target_feature_override = (
        Path(ARTIFACT_ROOT)
        / "laguna-dspark-conformance-20260814-v2"
        / "deepspec"
        / "raw_target_features.npy"
    )
    root.parent.mkdir(parents=True, exist_ok=True)
    telemetry, telemetry_output = _start_telemetry(root.parent / "vllm-gpu.csv")
    try:
        result = run_capture(
            output_root=root,
            prompt=prompt,
            target_feature_override=target_feature_override,
        )
    finally:
        telemetry.terminate()
        telemetry.wait(timeout=30)
        telemetry_output.close()
        artifacts.commit()
    return result


@app.function(image=vllm_image, **FUNCTION_OPTIONS)
def profile_vllm_production(
    run_id: str, prompt: str = DEFAULT_PROMPT
) -> dict[str, object]:
    from opjax.remote.laguna_vllm_conformance import run_production_profile

    root = Path(ARTIFACT_ROOT) / run_id / "production"
    root.parent.mkdir(parents=True, exist_ok=True)
    telemetry, telemetry_output = _start_telemetry(root.parent / "production-gpu.csv")
    try:
        result = run_production_profile(output_root=root, prompt=prompt)
    finally:
        telemetry.terminate()
        telemetry.wait(timeout=30)
        telemetry_output.close()
        artifacts.commit()
    return result


@app.local_entrypoint()
def main(run_id: str = "", prompt: str = DEFAULT_PROMPT) -> None:
    resolved_run_id = run_id or f"laguna-dspark-conformance-{uuid.uuid4().hex[:12]}"
    source = capture_deepspec.remote(resolved_run_id, prompt)
    adapter = capture_vllm.remote(resolved_run_id, prompt)
    print(
        json.dumps(
            {"run_id": resolved_run_id, "deepspec": source, "vllm": adapter},
            indent=2,
            sort_keys=True,
        )
    )
