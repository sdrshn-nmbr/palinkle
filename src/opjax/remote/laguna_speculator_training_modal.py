from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time

from huggingface_hub import snapshot_download
import modal
from safetensors import safe_open
import torch

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
serving_native = modal.Volume.from_name(
    "opjax-laguna-serving-native-v1",
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
    .add_local_file(
        "src/opjax/remote/build_laguna_serving_native_cache.py",
        "/opt/opjax-execution/build_laguna_serving_native_cache.py",
    )
    .add_local_file(
        "src/opjax/remote/prepare_laguna_serving_native_tokens.py",
        "/opt/opjax-execution/prepare_laguna_serving_native_tokens.py",
    )
    .add_local_dir(
        "data/pallas/corpora/laguna-speculator-v1",
        "/opt/opjax-data",
    )
)

VOLUMES = {
    HF_CACHE_DIR: cache,
    str(ROOT): training,
    "/mnt/serving-native": serving_native,
}
OPTIONS = {
    "image": image,
    "volumes": VOLUMES,
    "secrets": [secret],
    "timeout": 86_400,
}
TARGET_ID = "poolside/Laguna-XS-2.1"
TARGET_REVISION = "e9df9a59996d790b94b70f3fef343fe1d9e34bdf"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _checkpoint_identity(path: Path) -> dict[str, object]:
    files = {
        str(candidate.relative_to(path)): _sha256(candidate)
        for candidate in sorted(path.rglob("*"))
        if candidate.is_file()
        and candidate.name in {"config.json", "model.safetensors"}
    }
    if "config.json" not in files or "model.safetensors" not in files:
        raise RuntimeError(f"LAGUNA_CHECKPOINT_IDENTITY_INCOMPLETE:{path}")
    return {
        "path": str(path.resolve()),
        "files": files,
        "sha256": hashlib.sha256(
            json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    }


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _validated_json(path: Path, hash_key: str) -> dict[str, object]:
    if not path.is_file():
        raise RuntimeError(f"LAGUNA_BOUND_JSON_MISSING:{path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    claimed = value.get(hash_key)
    computed = _canonical_sha256(
        {key: item for key, item in value.items() if key != hash_key}
    )
    if claimed != computed:
        raise RuntimeError(f"LAGUNA_BOUND_JSON_HASH_INVALID:{path}:{claimed}:{computed}")
    return value


def _experiment_root(namespace: str) -> Path:
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,63}", namespace):
        raise ValueError(f"LAGUNA_TRAINING_NAMESPACE_INVALID:{namespace}")
    return ROOT / "experiments" / namespace


@app.function(image=image, volumes=VOLUMES, secrets=[secret], timeout=3600, memory=8192)
def audit_dflash_head(step: int) -> dict[str, object]:
    checkpoint = ROOT / "checkpoints" / "dflash" / f"step_{step}"
    index_root = Path(
        snapshot_download(
            TARGET_ID,
            revision=TARGET_REVISION,
            allow_patterns=["model.safetensors.index.json"],
        )
    )
    index = json.loads(
        (index_root / "model.safetensors.index.json").read_text(encoding="utf-8")
    )
    shard_name = index["weight_map"]["lm_head.weight"]
    target_root = Path(
        snapshot_download(
            TARGET_ID,
            revision=TARGET_REVISION,
            allow_patterns=["model.safetensors.index.json", shard_name],
        )
    )
    with safe_open(
        target_root / shard_name, framework="pt", device="cpu"
    ) as target_handle:
        target = target_handle.get_tensor("lm_head.weight")
    with safe_open(
        checkpoint / "model.safetensors", framework="pt", device="cpu"
    ) as checkpoint_handle:
        candidate = checkpoint_handle.get_tensor("lm_head.weight")
    if target.shape != candidate.shape or target.dtype != candidate.dtype:
        raise RuntimeError(
            f"LAGUNA_DFLASH_HEAD_SHAPE_INVALID:{target.shape}:{candidate.shape}:"
            f"{target.dtype}:{candidate.dtype}"
        )
    payload: dict[str, object] = {
        "schema_version": 1,
        "step": step,
        "target_revision": TARGET_REVISION,
        "target_shard": shard_name,
        "target_shard_sha256": _sha256(target_root / shard_name),
        "checkpoint_sha256": _sha256(checkpoint / "model.safetensors"),
        "shape": list(target.shape),
        "dtype": str(target.dtype),
        "exact_equal": bool(torch.equal(target, candidate)),
        "max_abs_error": float((target.float() - candidate.float()).abs().max()),
    }
    output = ROOT / "runs" / "audit" / "dflash" / f"step_{step}"
    output.mkdir(parents=True, exist_ok=True)
    (output / "head-parity.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    training.commit()
    return payload


def _execution_hashes() -> dict[str, str]:
    paths = {
        "native_tools_patch": Path("/opt/deepspec-native-tools.patch"),
        "dspark_config": CONFIG_ROOT / "laguna-dspark-training.py",
        "dflash_config": CONFIG_ROOT / "laguna-dflash-training.py",
        "corpus_manifest": DATA_ROOT / "manifest.json",
        "training_driver": Path(__file__),
        "serving_native_cache_builder": Path(
            "/opt/opjax-execution/build_laguna_serving_native_cache.py"
        ),
        "serving_native_token_builder": Path(
            "/opt/opjax-execution/prepare_laguna_serving_native_tokens.py"
        ),
    }
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise RuntimeError(f"LAGUNA_EXECUTION_FILES_MISSING:{','.join(missing)}")
    return {name: _sha256(path) for name, path in paths.items()}


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


def _run_observed(
    command: list[str],
    run_root: Path,
    *,
    environment: dict[str, str] | None = None,
) -> dict[str, object]:
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
                env=environment,
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
        "execution_files": _execution_hashes(),
        "deepspec_diff_sha256": hashlib.sha256(
            subprocess.check_output(["git", "-C", str(DEEPSPEC), "diff", "--binary"])
        ).hexdigest(),
    }
    (run_root / "runtime.json").write_text(
        json.dumps(runtime, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return runtime


@app.function(
    image=image, volumes=VOLUMES, secrets=[secret], timeout=3600, memory=16384
)
def initialize(arm: str, namespace: str = "legacy") -> dict[str, object]:
    experiment_root = _experiment_root(namespace)
    output = experiment_root / "initialized" / arm
    if output.exists():
        return json.loads((output / "initialization.json").read_text(encoding="utf-8"))
    command = [
        "python",
        "-m",
        "opjax.remote.initialize_laguna_speculators",
        "--arm",
        arm,
        "--output-root",
        str(experiment_root / "initialized"),
    ]
    run_root = experiment_root / "runs" / "initialize" / arm
    run_root.mkdir(parents=True, exist_ok=True)
    with (run_root / "run.log").open("wb") as log:
        subprocess.run(
            command, cwd=DEEPSPEC, check=True, stdout=log, stderr=subprocess.STDOUT
        )
    training.commit()
    return json.loads((output / "initialization.json").read_text(encoding="utf-8"))


@app.function(image=image, volumes=VOLUMES, timeout=1800, memory=16_384)
def audit_initialization_pair(
    namespace: str = "serving-native-v2",
) -> dict[str, object]:
    experiment_root = _experiment_root(namespace)
    paths = {
        arm: experiment_root / "initialized" / arm / "model.safetensors"
        for arm in ("dflash", "dspark")
    }
    if not all(path.is_file() for path in paths.values()):
        raise RuntimeError("LAGUNA_INITIALIZATION_PAIR_MISSING")
    with safe_open(paths["dflash"], framework="pt", device="cpu") as dflash, safe_open(
        paths["dspark"], framework="pt", device="cpu"
    ) as dspark:
        dflash_keys = set(dflash.keys())
        dspark_keys = set(dspark.keys())
        extra = dspark_keys - dflash_keys
        expected_extra = {
            "markov_head.markov_w1.weight",
            "markov_head.markov_w2.weight",
            "confidence_head.proj.weight",
            "confidence_head.proj.bias",
        }
        if extra != expected_extra or dflash_keys - dspark_keys:
            raise RuntimeError(
                f"LAGUNA_INITIALIZATION_PAIR_KEYS_INVALID:{sorted(extra)}:"
                f"{sorted(dflash_keys - dspark_keys)}"
            )
        mismatches = [
            name
            for name in sorted(dflash_keys)
            if not torch.equal(dflash.get_tensor(name), dspark.get_tensor(name))
        ]
    if mismatches:
        raise RuntimeError(f"LAGUNA_INITIALIZATION_PAIR_MISMATCH:{mismatches[:8]}")
    result: dict[str, object] = {
        "schema_version": 1,
        "namespace": namespace,
        "shared_tensor_count": len(dflash_keys),
        "shared_tensors_exact": True,
        "dspark_only_tensors": sorted(expected_extra),
        "checkpoints": {arm: _sha256(path) for arm, path in paths.items()},
    }
    result["sha256"] = hashlib.sha256(
        json.dumps(result, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    output = experiment_root / "runs" / "initialize" / "pair-audit.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    training.commit()
    return result


@app.function(image=image, volumes=VOLUMES, timeout=1800, memory=16_384)
def freeze_training_preflight(
    namespace: str = "serving-native-v2",
    capture_run_id: str = "serving-native-v2",
    supersede_existing: bool = False,
) -> dict[str, object]:
    experiment_root = _experiment_root(namespace)
    output = experiment_root / "training-preflight.json"
    capture_release_path = Path("/mnt/serving-native") / capture_run_id / "release.json"
    capture_release = _validated_json(capture_release_path, "release_sha256")
    pair_path = experiment_root / "runs" / "initialize" / "pair-audit.json"
    pair = _validated_json(pair_path, "sha256")
    caches = {}
    ordered_ids = {}
    for split in ("train", "calibration", "heldout"):
        cache_root = experiment_root / "cache" / split
        receipt_path = cache_root / "opjax-receipt.json"
        cache_manifest_path = cache_root / "manifest.json"
        receipt = _validated_json(receipt_path, "receipt_sha256")
        cache_manifest = json.loads(cache_manifest_path.read_text(encoding="utf-8"))
        source_samples = cache_manifest.get("source_samples")
        if (
            receipt.get("capture_release_sha256")
            != capture_release["release_sha256"]
            or not isinstance(source_samples, list)
            or len(source_samples) != capture_release["splits"][split]["record_count"]
        ):
            raise RuntimeError(f"LAGUNA_PREFLIGHT_CACHE_INVALID:{split}")
        ids = [sample["prompt_id"] for sample in source_samples]
        expected_ids = [record["id"] for record in capture_release["splits"][split]["records"]]
        if ids != expected_ids:
            raise RuntimeError(f"LAGUNA_PREFLIGHT_CACHE_ORDER_INVALID:{split}")
        ordered_ids[split] = ids
        caches[split] = {
            "receipt_file_sha256": _sha256(receipt_path),
            "receipt_sha256": receipt["receipt_sha256"],
            "manifest_file_sha256": _sha256(cache_manifest_path),
            "sample_count": len(ids),
            "ordered_ids_sha256": _canonical_sha256(ids),
        }
    initializations = {}
    for arm in ("dflash", "dspark"):
        root = experiment_root / "initialized" / arm
        manifest_path = root / "initialization.json"
        initializations[arm] = {
            "manifest_file_sha256": _sha256(manifest_path),
            "checkpoint": _checkpoint_identity(root),
        }
    value: dict[str, object] = {
        "schema_version": 1,
        "kind": "opjax_laguna_serving_native_training_preflight",
        "namespace": namespace,
        "capture_run_id": capture_run_id,
        "capture_release_sha256": capture_release["release_sha256"],
        "capture_release_file_sha256": _sha256(capture_release_path),
        "caches": caches,
        "ordered_ids": ordered_ids,
        "initializations": initializations,
        "pair_audit_sha256": pair["sha256"],
        "pair_audit_file_sha256": _sha256(pair_path),
        "configs": {
            arm: _sha256(CONFIG_ROOT / f"laguna-{arm}-training.py")
            for arm in ("dflash", "dspark")
        },
        "seed": 42,
        "global_batch_size": 8,
        "epochs": 10,
        "execution_files": _execution_hashes(),
        "deepspec_revision": DEEPSPEC_REVISION,
    }
    value["preflight_sha256"] = _canonical_sha256(value)
    if output.exists():
        existing = _validated_json(output, "preflight_sha256")
        if existing != value:
            if not supersede_existing:
                raise RuntimeError("LAGUNA_TRAINING_PREFLIGHT_DRIFT")
            archive = experiment_root / "training-preflight-attempts" / str(
                time.time_ns()
            )
            archive.parent.mkdir(parents=True, exist_ok=True)
            os.replace(output, archive)
        else:
            return existing
    output.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    training.commit()
    return value


@app.function(image=image, volumes=VOLUMES, timeout=3600, memory=8192)
def audit_checkpoint_lineage(
    namespace: str = "serving-native-v2",
) -> dict[str, object]:
    experiment_root = _experiment_root(namespace)
    preflight = _validated_json(
        experiment_root / "training-preflight.json", "preflight_sha256"
    )
    steps = (0, 13, 26, 39, 52, 65, 78, 91, 104, 117, 120)
    checkpoints: dict[str, dict[str, object]] = {}
    training_results: dict[str, object] = {}
    for arm in ("dflash", "dspark"):
        arm_checkpoints = {}
        for step in steps:
            root = (
                experiment_root / "initialized" / arm
                if step == 0
                else experiment_root / "checkpoints" / arm / f"step_{step}"
            )
            arm_checkpoints[str(step)] = _checkpoint_identity(root)
        if arm_checkpoints["0"] != preflight["initializations"][arm]["checkpoint"]:
            raise RuntimeError(f"LAGUNA_LINEAGE_INITIALIZATION_MISMATCH:{arm}")
        result_path = experiment_root / "runs" / "train" / arm / "result.json"
        train_result = _validated_json(result_path, "result_sha256")
        if (
            train_result.get("preflight_sha256") != preflight["preflight_sha256"]
            or train_result.get("checkpoint") != arm_checkpoints["120"]
        ):
            raise RuntimeError(f"LAGUNA_LINEAGE_FINAL_MISMATCH:{arm}")
        checkpoints[arm] = arm_checkpoints
        training_results[arm] = {
            "result_sha256": train_result["result_sha256"],
            "result_file_sha256": _sha256(result_path),
        }
    value: dict[str, object] = {
        "schema_version": 1,
        "kind": "opjax_laguna_checkpoint_lineage",
        "namespace": namespace,
        "preflight_sha256": preflight["preflight_sha256"],
        "steps": list(steps),
        "checkpoints": checkpoints,
        "training_results": training_results,
    }
    value["sha256"] = _canonical_sha256(value)
    output = experiment_root / "runs" / "eval" / "checkpoint-lineage.json"
    output.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    training.commit()
    return value


@app.function(image=image, gpu="T4", volumes=VOLUMES, timeout=7200, memory=32_768)
def prepare_serving_native_tokens(
    capture_run_id: str = "serving-native-v2",
) -> dict[str, object]:
    output = Path("/mnt/serving-native") / capture_run_id / "tokenized"
    run_root = _experiment_root(capture_run_id) / "runs" / "tokenize"
    if output.exists():
        archive = output.parent / "tokenized-attempts" / str(time.time_ns())
        archive.parent.mkdir(parents=True, exist_ok=True)
        os.replace(output, archive)
    if run_root.exists():
        archive = run_root.parent / "tokenize-attempts" / str(time.time_ns())
        archive.parent.mkdir(parents=True, exist_ok=True)
        os.replace(run_root, archive)
    runtime = _run_observed(
        [
            "python",
            "-m",
            "opjax.remote.prepare_laguna_serving_native_tokens",
            "--source-root",
            str(DATA_ROOT),
            "--output",
            str(output),
        ],
        run_root,
    )
    release = json.loads((output / "release.json").read_text(encoding="utf-8"))
    result = {**release, "runtime": runtime}
    (run_root / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    serving_native.commit()
    training.commit()
    return result


@app.function(image=image, gpu="T4", volumes=VOLUMES, timeout=7200, memory=65_536)
def build_serving_native_cache(
    split: str,
    namespace: str = "serving-native-v2",
    capture_run_id: str = "serving-native-v2",
    supersede_existing: bool = False,
) -> dict[str, object]:
    if split not in {"train", "calibration", "heldout"}:
        raise ValueError(f"LAGUNA_CACHE_SPLIT_INVALID:{split}")
    experiment_root = _experiment_root(namespace)
    output = experiment_root / "cache" / split
    run_root = experiment_root / "runs" / "cache" / split
    if output.exists():
        receipt_path = output / "opjax-receipt.json"
        if receipt_path.is_file() and not supersede_existing:
            raise RuntimeError(f"LAGUNA_CACHE_ALREADY_COMPLETE:{split}")
        archive = experiment_root / "cache-attempts" / split / str(time.time_ns())
        archive.parent.mkdir(parents=True, exist_ok=True)
        os.replace(output, archive)
    if run_root.exists():
        archive = (
            experiment_root / "runs" / "cache-attempts" / split / str(time.time_ns())
        )
        archive.parent.mkdir(parents=True, exist_ok=True)
        os.replace(run_root, archive)
    runtime = _run_observed(
        [
            "python",
            "-m",
            "opjax.remote.build_laguna_serving_native_cache",
            "--capture-root",
            str(Path("/mnt/serving-native") / capture_run_id),
            "--split",
            split,
            "--output",
            str(output),
        ],
        run_root,
    )
    receipt = json.loads((output / "opjax-receipt.json").read_text(encoding="utf-8"))
    receipt["runtime"] = runtime
    (run_root / "result.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    training.commit()
    return receipt


@app.function(gpu="H200:4", **OPTIONS)
def prepare_cache(split: str) -> dict[str, object]:
    if split not in {"train", "calibration", "heldout"}:
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


@app.function(gpu="H200:4", **OPTIONS)
def train_arm(
    arm: str,
    namespace: str = "legacy",
    supersede_failed_attempt: bool = False,
) -> dict[str, object]:
    if arm not in {"dflash", "dspark"}:
        raise ValueError(f"LAGUNA_TRAIN_ARM_INVALID:{arm}")
    config = CONFIG_ROOT / f"laguna-{arm}-training.py"
    experiment_root = _experiment_root(namespace)
    preflight_path = experiment_root / "training-preflight.json"
    preflight = _validated_json(preflight_path, "preflight_sha256")
    if preflight.get("namespace") != namespace:
        raise RuntimeError(f"LAGUNA_TRAIN_PREFLIGHT_NAMESPACE:{namespace}")
    run_root = experiment_root / "runs" / "train" / arm
    result_path = run_root / "result.json"
    if result_path.is_file():
        result = _validated_json(result_path, "result_sha256")
        if result.get("preflight_sha256") != preflight["preflight_sha256"]:
            raise RuntimeError(f"LAGUNA_TRAIN_RESULT_PREFLIGHT:{arm}")
        return result
    checkpoint_root = experiment_root / "checkpoints" / arm
    if run_root.exists() or checkpoint_root.exists():
        if not supersede_failed_attempt:
            raise RuntimeError(f"LAGUNA_TRAIN_PARTIAL_REQUIRES_EXPLICIT_RESUME:{arm}")
        attempt_root = (
            experiment_root / "runs" / "train-attempts" / arm / str(time.time_ns())
        )
        attempt_root.mkdir(parents=True, exist_ok=False)
        if run_root.exists():
            os.replace(run_root, attempt_root / "run")
        if checkpoint_root.exists():
            os.replace(checkpoint_root, attempt_root / "checkpoints")
    environment = {**os.environ, "OPJAX_LAGUNA_TRAINING_NAMESPACE": namespace}
    runtime = _run_observed(
        ["python", "train.py", "--config", str(config)],
        run_root,
        environment=environment,
    )
    latest = (experiment_root / "checkpoints" / arm / "step_latest").resolve()
    result: dict[str, object] = {
        "schema_version": 1,
        "arm": arm,
        "namespace": namespace,
        "preflight_sha256": preflight["preflight_sha256"],
        "checkpoint": _checkpoint_identity(latest),
        "runtime": runtime,
    }
    result["result_sha256"] = _canonical_sha256(result)
    result_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    training.commit()
    return result


@app.function(gpu="H200", **OPTIONS)
def evaluate_arm(
    arm: str,
    step: int,
    split: str = "calibration",
    variant: str = "raw",
    namespace: str = "legacy",
) -> dict[str, object]:
    if arm not in {"dflash", "dspark"}:
        raise ValueError(f"LAGUNA_EVAL_ARM_INVALID:{arm}")
    if split not in {"calibration", "heldout"}:
        raise ValueError(f"LAGUNA_EVAL_SPLIT_INVALID:{split}")
    if variant not in {"raw", "calibrated"}:
        raise ValueError(f"LAGUNA_EVAL_VARIANT_INVALID:{variant}")
    if variant == "calibrated" and (arm != "dspark" or step == 0):
        raise ValueError(f"LAGUNA_EVAL_VARIANT_UNSUPPORTED:{arm}:{step}:{variant}")
    experiment_root = _experiment_root(namespace)
    if variant == "calibrated":
        checkpoint = experiment_root / "calibrated" / "dspark" / f"step_{step}"
    else:
        checkpoint = (
            experiment_root / "initialized" / arm
            if step == 0
            else experiment_root / "checkpoints" / arm / f"step_{step}"
        )
    run_root = (
        experiment_root / "runs" / "eval" / split / arm / variant / f"step_{step}"
    )
    command = [
        "python",
        "-m",
        "opjax.remote.evaluate_laguna_speculator",
        "--checkpoint",
        str(checkpoint),
        "--cache",
        str(experiment_root / "cache" / split),
        "--output",
        str(run_root),
        "--num-anchors",
        "64",
        "--seed",
        "42",
    ]
    runtime = _run_observed(command, run_root)
    payload = json.loads((run_root / "evaluation.json").read_text(encoding="utf-8"))
    payload["arm"] = arm
    payload["step"] = step
    payload["split"] = split
    payload["variant"] = variant
    payload["checkpoint"] = _checkpoint_identity(checkpoint)
    payload["runtime"] = runtime
    (run_root / "result.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    training.commit()
    return payload


@app.function(
    image=image, volumes=VOLUMES, secrets=[secret], timeout=3600, memory=16384
)
def export_dflash(step: int, namespace: str = "legacy") -> dict[str, object]:
    experiment_root = _experiment_root(namespace)
    output = experiment_root / "exports" / "dflash" / f"step_{step}"
    command = [
        "python",
        "-m",
        "opjax.remote.export_laguna_speculator",
        "--checkpoint",
        str(experiment_root / "checkpoints" / "dflash" / f"step_{step}"),
        "--output",
        str(output),
    ]
    run_root = experiment_root / "runs" / "export" / "dflash" / f"step_{step}"
    run_root.mkdir(parents=True, exist_ok=True)
    with (run_root / "run.log").open("wb") as log:
        subprocess.run(
            command, cwd=DEEPSPEC, check=True, stdout=log, stderr=subprocess.STDOUT
        )
    training.commit()
    return json.loads((output / "export.json").read_text(encoding="utf-8"))


@app.function(gpu="H200", **OPTIONS)
def calibrate_dspark(step: int, namespace: str = "legacy") -> dict[str, object]:
    experiment_root = _experiment_root(namespace)
    output = experiment_root / "calibrated" / "dspark" / f"step_{step}"
    command = [
        "python",
        "-m",
        "opjax.remote.calibrate_laguna_dspark",
        "--checkpoint",
        str(experiment_root / "checkpoints" / "dspark" / f"step_{step}"),
        "--cache",
        str(experiment_root / "cache" / "calibration"),
        "--output",
        str(output),
    ]
    run_root = experiment_root / "runs" / "calibration" / "dspark" / f"step_{step}"
    runtime = _run_observed(command, run_root)
    payload = json.loads((output / "calibration.json").read_text(encoding="utf-8"))
    payload["runtime"] = runtime
    (run_root / "result.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    training.commit()
    return payload


@app.function(image=image, volumes=VOLUMES, timeout=300, memory=4096)
def select_checkpoint(
    arm: str, step: int, namespace: str = "legacy"
) -> dict[str, object]:
    if arm not in {"dflash", "dspark"}:
        raise ValueError(f"LAGUNA_SELECT_ARM_INVALID:{arm}")
    experiment_root = _experiment_root(namespace)
    evaluation = (
        experiment_root
        / "runs"
        / "eval"
        / "calibration"
        / arm
        / "raw"
        / f"step_{step}"
        / "result.json"
    )
    if not evaluation.is_file():
        raise RuntimeError(f"LAGUNA_SELECT_EVALUATION_MISSING:{evaluation}")
    source = (
        experiment_root / "exports" / "dflash" / f"step_{step}"
        if arm == "dflash"
        else experiment_root / "calibrated" / "dspark" / f"step_{step}"
    )
    identity = _checkpoint_identity(source)
    selected_root = experiment_root / "selected"
    selected_root.mkdir(parents=True, exist_ok=True)
    destination = selected_root / arm
    temporary = selected_root / f".{arm}.tmp"
    temporary.unlink(missing_ok=True)
    temporary.symlink_to(source, target_is_directory=True)
    temporary.replace(destination)
    payload = {
        "schema_version": 1,
        "arm": arm,
        "step": step,
        "evaluation_sha256": _sha256(evaluation),
        "checkpoint": identity,
    }
    payload["sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    (selected_root / f"{arm}.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    training.commit()
    return payload


@app.function(image=image, volumes=VOLUMES, secrets=[secret], timeout=300)
def audit_inputs() -> dict[str, object]:
    result: dict[str, object] = {
        "schema_version": 1,
        "deepspec_revision": DEEPSPEC_REVISION,
        "execution_files": _execution_hashes(),
        "deepspec_diff_sha256": hashlib.sha256(
            subprocess.check_output(["git", "-C", str(DEEPSPEC), "diff", "--binary"])
        ).hexdigest(),
    }
    output = ROOT / "runs" / "input-audit.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    training.commit()
    return result
