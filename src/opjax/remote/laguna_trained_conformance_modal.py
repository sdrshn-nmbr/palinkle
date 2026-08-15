from __future__ import annotations

import json
from pathlib import Path
import subprocess

from huggingface_hub import snapshot_download
import modal

from opjax.pallas.laguna_dspark_conformance import (
    build_conformance_report,
    build_dflash_conformance_report,
    build_target_feature_conformance_report,
    validate_dflash_conformance_report,
    validate_target_feature_conformance_report,
)
from opjax.pallas.laguna_speculative import TARGET_ID, TARGET_REVISION
from opjax.remote.config import (
    HF_CACHE_DIR,
    HF_CACHE_VOLUME_NAME,
    MODAL_ENVIRONMENT,
    MODAL_SECRET_NAME,
    MODAL_VOLUME_VERSION,
    REMOTE_ENV,
)
from opjax.remote.laguna_vllm_conformance import run_capture, run_dflash_capture


APP_NAME = "opjax-laguna-trained-conformance-v1"
ROOT = Path("/mnt/training")
ARTIFACT_ROOT = Path("/mnt/conformance")
DEEPSPEC_REVISION = "787db11ea347ac3944233e5aa9c7f1bd8a9b5ced"
PROMPT = "Use the bash tool to print the current working directory."

app = modal.App(APP_NAME)
cache = modal.Volume.from_name(
    HF_CACHE_VOLUME_NAME,
    environment_name=MODAL_ENVIRONMENT,
    create_if_missing=True,
    version=MODAL_VOLUME_VERSION,
)
training = modal.Volume.from_name(
    "opjax-laguna-speculator-training-v1",
    environment_name=MODAL_ENVIRONMENT,
    create_if_missing=True,
    version=MODAL_VOLUME_VERSION,
)
artifacts = modal.Volume.from_name(
    "opjax-laguna-trained-conformance-v1",
    environment_name=MODAL_ENVIRONMENT,
    create_if_missing=True,
    version=MODAL_VOLUME_VERSION,
)
secret = modal.Secret.from_name(MODAL_SECRET_NAME, environment_name=MODAL_ENVIRONMENT)

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
            "OPJAX_SPEC_ARTIFACT_ROOT": str(ARTIFACT_ROOT / "runtime"),
            "OPJAX_SPEC_ARTIFACT_VOLUME": "opjax-laguna-trained-conformance-v1",
            "OPJAX_SPEC_MODAL_ENVIRONMENT": MODAL_ENVIRONMENT,
        }
    )
    .add_local_python_source("opjax")
)
VOLUMES = {
    HF_CACHE_DIR: cache,
    str(ROOT): training,
    str(ARTIFACT_ROOT): artifacts,
}
OPTIONS = {"volumes": VOLUMES, "secrets": [secret], "timeout": 3600}


def _target_path() -> Path:
    return Path(
        snapshot_download(
            TARGET_ID,
            revision=TARGET_REVISION,
            local_dir="/tmp/opjax-laguna-target",
        )
    )


@app.function(image=deepspec_image, gpu="H200", **OPTIONS)
def capture_source(
    run_id: str, arm: str, prompt: str = PROMPT
) -> dict[str, object]:
    if arm not in {"dflash", "dspark"}:
        raise ValueError(f"LAGUNA_CONFORMANCE_ARM_INVALID:{arm}")
    output = ARTIFACT_ROOT / run_id / "source"
    selection = json.loads((ROOT / "selected" / f"{arm}.json").read_text())
    draft = (
        ROOT / "checkpoints" / "dflash" / f"step_{selection['step']}"
        if arm == "dflash"
        else ROOT / "selected" / "dspark"
    )
    module = (
        "opjax.remote.laguna_deepspec_dflash_conformance"
        if arm == "dflash"
        else "opjax.remote.laguna_deepspec_conformance"
    )
    command = [
        "python",
        "-m",
        module,
        "--output-root",
        str(output),
        "--prompt",
        prompt,
        "--target-path",
        str(_target_path()),
        "--draft-path",
        str(draft),
    ]
    log = ARTIFACT_ROOT / run_id / "source.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("wb") as handle:
        subprocess.run(
            command,
            cwd="/opt/deepspec",
            check=True,
            stdout=handle,
            stderr=subprocess.STDOUT,
        )
    artifacts.commit()
    return json.loads((output / "manifest.json").read_text())


@app.function(image=vllm_image, gpu="H200", **OPTIONS)
def capture_adapter(run_id: str, arm: str) -> dict[str, object]:
    if arm not in {"dflash", "dspark"}:
        raise ValueError(f"LAGUNA_CONFORMANCE_ARM_INVALID:{arm}")
    source = ARTIFACT_ROOT / run_id / "source"
    output = ARTIFACT_ROOT / run_id / "adapter"
    kwargs = {
        "output_root": output,
        "prompt": PROMPT,
        "target_feature_override": source / "raw_target_features.npy",
        "draft_model": str(ROOT / "selected" / arm),
    }
    result = run_dflash_capture(**kwargs) if arm == "dflash" else run_capture(**kwargs)
    artifacts.commit()
    return result


@app.function(image=vllm_image, gpu="H200", **OPTIONS)
def capture_adapter_live(
    run_id: str, prompt: str = PROMPT, target_features_only: bool = False
) -> dict[str, object]:
    output = ARTIFACT_ROOT / run_id / (
        "adapter-live-target-only" if target_features_only else "adapter-live"
    )
    result = run_capture(
        output_root=output,
        prompt=prompt,
        target_feature_override=None,
        draft_model=str(ROOT / "selected" / "dspark"),
        target_features_only=target_features_only,
    )
    artifacts.commit()
    return result


@app.function(image=deepspec_image, **OPTIONS)
def compare(run_id: str, arm: str) -> dict[str, object]:
    root = ARTIFACT_ROOT / run_id
    source_root = root / "source"
    adapter_root = root / "adapter"
    source = json.loads((source_root / "manifest.json").read_text())
    adapter = json.loads((adapter_root / "manifest.json").read_text())
    if arm == "dflash":
        report = build_dflash_conformance_report(
            source_root=source_root,
            source_capture=source,
            adapter_root=adapter_root,
            adapter_capture=adapter,
        )
        validate_dflash_conformance_report(report, root=root)
    else:
        report = build_conformance_report(
            source_root=source_root,
            source_capture=source,
            adapter_root=adapter_root,
            adapter_capture=adapter,
            mutation_controls=source["mutation_controls"],
        )
    (root / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    artifacts.commit()
    return report


@app.function(image=deepspec_image, **OPTIONS)
def compare_live(run_id: str) -> dict[str, object]:
    root = ARTIFACT_ROOT / run_id
    source_root = root / "source"
    adapter_root = root / "adapter-live"
    source = json.loads((source_root / "manifest.json").read_text())
    adapter = json.loads((adapter_root / "manifest.json").read_text())
    report = build_conformance_report(
        source_root=source_root,
        source_capture=source,
        adapter_root=adapter_root,
        adapter_capture=adapter,
        mutation_controls=source["mutation_controls"],
    )
    (root / "report-live.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    target_feature_report = build_target_feature_conformance_report(
        source_root=source_root,
        source_capture=source,
        adapter_root=adapter_root,
        adapter_capture=adapter,
    )
    validate_target_feature_conformance_report(
        target_feature_report, root=root, require_pass=False
    )
    (root / "target-feature-report.json").write_text(
        json.dumps(target_feature_report, indent=2, sort_keys=True) + "\n"
    )
    artifacts.commit()
    return {
        "downstream_without_override": report,
        "target_features": target_feature_report,
    }


@app.function(image=deepspec_image, **OPTIONS)
def compare_target_features(run_id: str) -> dict[str, object]:
    root = ARTIFACT_ROOT / run_id
    source_root = root / "source"
    adapter_root = root / "adapter-live-target-only"
    source = json.loads((source_root / "manifest.json").read_text())
    adapter = json.loads((adapter_root / "manifest.json").read_text())
    report = build_target_feature_conformance_report(
        source_root=source_root,
        source_capture=source,
        adapter_root=adapter_root,
        adapter_capture=adapter,
    )
    validate_target_feature_conformance_report(report, root=root, require_pass=False)
    (root / "target-feature-report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    artifacts.commit()
    return report
