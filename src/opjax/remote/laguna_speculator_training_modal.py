from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import time

import modal

from opjax.remote.config import (
    HF_CACHE_DIR,
    HF_CACHE_VOLUME_NAME,
    MODAL_ENVIRONMENT,
    MODAL_SECRET_NAME,
    MODAL_VOLUME_VERSION,
    REMOTE_ENV,
)


APP_NAME = "opjax-laguna-speculator-training-v1"
DEEPSPEC_REVISION = "787db11ea347ac3944233e5aa9c7f1bd8a9b5ced"
ROOT = Path("/mnt/training")
DEEPSPEC = Path("/opt/deepspec")
CONFIG_ROOT = Path("/opt/opjax-config")
DATA_ROOT = Path("/opt/opjax-data")

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
secret = modal.Secret.from_name(MODAL_SECRET_NAME, environment_name=MODAL_ENVIRONMENT)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .uv_pip_install(
        "torch==2.9.1",
        "transformers==5.10.2",
        "numpy==2.4.4",
        "PyYAML==6.0.3",
        "tqdm==4.67.3",
        "tensorboard==2.20.0",
        "matplotlib==3.10.9",
        "triton==3.5.1",
        "typing_extensions==4.15.0",
        "sentencepiece==0.2.1",
        "safetensors==0.7.0",
        "prettytable==3.17.0",
        "datasets==4.8.5",
        "compressed-tensors==0.15.0.1",
        "psutil==7.2.2",
        "accelerate==1.14.0",
        "huggingface-hub==1.5.0",
    )
    .run_commands(
        "git clone https://github.com/RespectMathias/DeepSpec.git /opt/deepspec",
        f"git -C /opt/deepspec checkout {DEEPSPEC_REVISION}",
    )
    .add_local_file(
        "scripts/pallas/deepspec_native_tools.patch",
        "/opt/deepspec-native-tools.patch",
        copy=True,
    )
    .run_commands(
        "git -C /opt/deepspec apply --check /opt/deepspec-native-tools.patch",
        "git -C /opt/deepspec apply /opt/deepspec-native-tools.patch",
    )
    .env({**REMOTE_ENV, "PYTHONPATH": "/opt/deepspec"})
    .add_local_python_source("opjax")
    .add_local_file(
        "config/pallas/laguna-dspark-training.py",
        "/opt/opjax-config/laguna-dspark-training.py",
    )
    .add_local_file(
        "config/pallas/laguna-dflash-training.py",
        "/opt/opjax-config/laguna-dflash-training.py",
    )
    .add_local_dir(
        "data/pallas/corpora/laguna-speculator-v1",
        "/opt/opjax-data",
    )
)

VOLUMES = {HF_CACHE_DIR: cache, str(ROOT): training}
OPTIONS = {
    "image": image,
    "volumes": VOLUMES,
    "secrets": [secret],
    "timeout": 86_400,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _telemetry(path: Path) -> tuple[subprocess.Popen[bytes], object]:
    output = path.open("wb")
    process = subprocess.Popen(
        [
            "nvidia-smi",
            "--query-gpu=timestamp,index,name,memory.used,memory.total,utilization.gpu,utilization.memory,power.draw,temperature.gpu",
            "--format=csv",
            "--loop=1",
        ],
        stdout=output,
        stderr=subprocess.STDOUT,
    )
    return process, output


def _run_observed(command: list[str], run_root: Path) -> dict[str, object]:
    run_root.mkdir(parents=True, exist_ok=True)
    started = time.time()
    telemetry, telemetry_output = _telemetry(run_root / "gpu.csv")
    try:
        with (run_root / "run.log").open("wb") as log:
            subprocess.run(
                command,
                cwd=DEEPSPEC,
                check=True,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
    finally:
        telemetry.terminate()
        telemetry.wait(timeout=30)
        telemetry_output.close()
    finished = time.time()
    runtime = {
        "command": command,
        "started_at_unix": started,
        "finished_at_unix": finished,
        "wall_seconds": finished - started,
        "deepspec_revision": DEEPSPEC_REVISION,
        "torch": subprocess.check_output(
            ["python", "-c", "import torch; print(torch.__version__)"], text=True
        ).strip(),
        "transformers": subprocess.check_output(
            ["python", "-c", "import transformers; print(transformers.__version__)"],
            text=True,
        ).strip(),
        "gpu": subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
            text=True,
        ).strip(),
    }
    (run_root / "runtime.json").write_text(
        json.dumps(runtime, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return runtime


@app.function(image=image, volumes=VOLUMES, secrets=[secret], timeout=3600, memory=16384)
def initialize(arm: str) -> dict[str, object]:
    output = ROOT / "initialized" / arm
    if output.exists():
        return json.loads((output / "initialization.json").read_text(encoding="utf-8"))
    command = [
        "python",
        "-m",
        "opjax.remote.initialize_laguna_speculators",
        "--arm",
        arm,
        "--output-root",
        str(ROOT / "initialized"),
    ]
    run_root = ROOT / "runs" / "initialize" / arm
    run_root.mkdir(parents=True, exist_ok=True)
    with (run_root / "run.log").open("wb") as log:
        subprocess.run(command, cwd=DEEPSPEC, check=True, stdout=log, stderr=subprocess.STDOUT)
    training.commit()
    return json.loads((output / "initialization.json").read_text(encoding="utf-8"))


@app.function(gpu="H200", **OPTIONS)
def prepare_cache(split: str) -> dict[str, object]:
    if split not in {"train", "heldout"}:
        raise ValueError(f"LAGUNA_CACHE_SPLIT_INVALID:{split}")
    output = ROOT / "cache" / split
    command = [
        "python",
        "scripts/data/prepare_target_cache.py",
        "--config",
        str(CONFIG_ROOT / "laguna-dspark-training.py"),
        "--train-data-path",
        str(DATA_ROOT / f"{split}.jsonl"),
        "--output-dir",
        str(output),
        "--local-batch-size",
        "1",
        "--num-workers",
        "2",
        "--resume",
        "--resume-checkpoint-interval",
        "10",
    ]
    runtime = _run_observed(command, ROOT / "runs" / "cache" / split)
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    manifest["runtime"] = runtime
    training.commit()
    return manifest


@app.function(gpu="H200", **OPTIONS)
def train_arm(arm: str) -> dict[str, object]:
    if arm not in {"dflash", "dspark"}:
        raise ValueError(f"LAGUNA_TRAIN_ARM_INVALID:{arm}")
    config = CONFIG_ROOT / f"laguna-{arm}-training.py"
    runtime = _run_observed(
        ["python", "train.py", "--config", str(config)],
        ROOT / "runs" / "train" / arm,
    )
    latest = (ROOT / "checkpoints" / arm / "step_latest").resolve()
    model = latest / "model.safetensors"
    result = {
        "arm": arm,
        "checkpoint": str(latest),
        "checkpoint_sha256": _sha256(model),
        "runtime": runtime,
    }
    (ROOT / "runs" / "train" / arm / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    training.commit()
    return result


@app.function(gpu="H200", **OPTIONS)
def evaluate_arm(arm: str, step: int) -> dict[str, object]:
    if arm not in {"dflash", "dspark"}:
        raise ValueError(f"LAGUNA_EVAL_ARM_INVALID:{arm}")
    checkpoint = ROOT / "checkpoints" / arm / f"step_{step}"
    run_root = ROOT / "runs" / "eval" / arm / f"step_{step}"
    command = [
        "python",
        "-m",
        "opjax.remote.evaluate_laguna_speculator",
        "--checkpoint",
        str(checkpoint),
        "--cache",
        str(ROOT / "cache" / "heldout"),
        "--output",
        str(run_root),
        "--num-anchors",
        "64",
    ]
    runtime = _run_observed(command, run_root)
    payload = json.loads((run_root / "evaluation.json").read_text(encoding="utf-8"))
    payload["arm"] = arm
    payload["step"] = step
    payload["runtime"] = runtime
    (run_root / "result.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    training.commit()
    return payload


@app.function(image=image, volumes=VOLUMES, secrets=[secret], timeout=3600, memory=16384)
def export_dflash(step: int) -> dict[str, object]:
    output = ROOT / "exports" / "dflash" / f"step_{step}"
    command = [
        "python",
        "-m",
        "opjax.remote.export_laguna_speculator",
        "--checkpoint",
        str(ROOT / "checkpoints" / "dflash" / f"step_{step}"),
        "--output",
        str(output),
    ]
    run_root = ROOT / "runs" / "export" / "dflash" / f"step_{step}"
    run_root.mkdir(parents=True, exist_ok=True)
    with (run_root / "run.log").open("wb") as log:
        subprocess.run(command, cwd=DEEPSPEC, check=True, stdout=log, stderr=subprocess.STDOUT)
    training.commit()
    return json.loads((output / "export.json").read_text(encoding="utf-8"))
