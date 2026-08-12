"""Live Miles and SGLang acceptance control for the Phase 1 foundation."""

import hashlib
import importlib.metadata
import json
import os
import platform
import re
import shutil
import subprocess
import textwrap
import time
import uuid
from pathlib import Path

import modal
import pyarrow as pa
import pyarrow.parquet as pq
import requests
import torch
from huggingface_hub import snapshot_download
from torch.distributed.checkpoint import FileSystemReader
from transformers import AutoModelForCausalLM, AutoTokenizer

from opjax.pallas.experiment_foundation import (
    ArtifactRole,
    CheckpointDraft,
    CheckpointIdentity,
    CheckpointValidation,
    FilesystemCheckpointStore,
    ModelArm,
    RuntimeBinding,
    RuntimeCursor,
    TrainingMethod,
)


APP_NAME = "opjax-phase1-miles-sglang-control"
MILES_IMAGE = "radixark/miles@sha256:08be00658cd24eaa364ca4ad0b1a3911dfbe4adc04fd0c148e4241402fb40812"
SGLANG_RUNTIME_IMAGE = (
    "lmsysorg/sglang@sha256:6fbf87218202af881b05cb6803bd47f56734727e18e4e4e6c358bc32ccdc0eac"
)
MOUNT = Path("/checkpoints")
RUN_ROOT = MOUNT / "phase1-live-control"
ACCEPTED_STORE_ROOT = MOUNT / "phase1-accepted-checkpoints"
HF_MOUNT = Path("/hf-cache")
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
MODEL_REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
MILES_REVISION = "b1860dd264e17c96d5d92da96c957d88cfd3a1f8"
SGLANG_REVISION = "c80a38edcd2c7077c909a5ed925c9241e754c067"
CONTROL_LANE = "sync-control"
INTERRUPTED_LANE = "sync-interrupted"
RESUMED_LANE = "sync-resumed"
PARITY_DTYPE = "float32"

app = modal.App(APP_NAME)
volume = modal.Volume.from_name("opjax-checkpoints-v2", create_if_missing=False)
hf_volume = modal.Volume.from_name("opjax-hf-cache-v2", create_if_missing=False)
secret = modal.Secret.from_name("opjax-secrets")
base_image = modal.Image.from_registry(MILES_IMAGE)
image = (
    base_image
    .add_local_dir("references/miles", "/opt/miles", copy=True)
    .add_local_dir("references/sglang/python", "/opt/sglang/python", copy=True)
    .add_local_python_source("opjax")
)
parity_image = (
    modal.Image.from_registry(SGLANG_RUNTIME_IMAGE)
    .add_local_dir("references/miles", "/opt/miles", copy=True)
    .add_local_dir("references/sglang/python", "/opt/sglang/python", copy=True)
    .add_local_python_source("opjax")
)


def _run(command: list[str]) -> str:
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )
    if completed.returncode:
        raise RuntimeError(
            "PHASE1_LIVE_COMMAND_FAILED: "
            f"command={command!r} exit={completed.returncode}\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    return completed.stdout


def _run_shell(script: str, *, timeout: int = 3600) -> str:
    completed = subprocess.run(
        ["bash", "-lc", script],
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
        env={**os.environ, "PYTHONUNBUFFERED": "1", "HF_HOME": str(HF_MOUNT)},
    )
    log = completed.stdout + "\n" + completed.stderr
    if completed.returncode:
        raise RuntimeError(
            "PHASE1_MILES_CONTROL_FAILED: "
            f"exit={completed.returncode}\n{log[-30000:]}"
        )
    return log


@app.function(
    image=image,
    gpu="H100",
    secrets=[secret],
    volumes={str(MOUNT): volume},
    timeout=600,
)
def preflight() -> dict[str, object]:
    probes = {
        "python": ["python", "--version"],
        "nvidia_smi": ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
        "miles": ["python", "-c", "import miles; print(list(miles.__path__))"],
        "sglang": ["python", "-c", "import sglang,inspect; print(inspect.getfile(sglang))"],
        "ray": ["python", "-c", "import ray; print(ray.__version__)"],
        "torch": ["python", "-c", "import torch; print(torch.__version__, torch.cuda.is_available())"],
    }
    result = {name: _run(command).strip() for name, command in probes.items()}
    result["image"] = MILES_IMAGE
    result["run_root"] = str(RUN_ROOT)
    return result


def _function_kwargs() -> dict[str, object]:
    return {
        "image": image,
        "gpu": "H100",
        "secrets": [secret],
        "volumes": {str(MOUNT): volume, str(HF_MOUNT): hf_volume},
        "timeout": 7200,
    }


@app.function(**_function_kwargs())
def prepare_control(run_id: str) -> dict[str, str]:
    root = RUN_ROOT / run_id
    root.mkdir(parents=True, exist_ok=False)
    model = Path(
        snapshot_download(
            repo_id=MODEL_ID,
            revision=MODEL_REVISION,
            cache_dir=HF_MOUNT,
        )
    )
    (root / "model-path.txt").write_text(str(model))
    messages = [
        [
            {"role": "user", "content": "Return the next integer after 10."},
            {"role": "assistant", "content": "11"},
        ],
        [
            {"role": "user", "content": "Return the next integer after 20."},
            {"role": "assistant", "content": "21"},
        ],
    ]
    pq.write_table(pa.table({"messages": messages}), root / "control.parquet")
    _run_shell(
        textwrap.dedent(
            f"""
            rm -rf {root}/base_torch_dist
            source /opt/miles/scripts/models/qwen2.5-0.5B.sh
            PYTHONPATH=/opt/miles:/root/Megatron-LM torchrun --nproc-per-node 1 \\
              /opt/miles/tools/convert_hf_to_torch_dist.py \\
              "${{MODEL_ARGS[@]}}" \\
              --hf-checkpoint {model} \\
              --save {root}/base_torch_dist
            """
        )
    )
    volume.commit()
    hf_volume.commit()
    return {"run_id": run_id, "model": str(model), "root": str(root)}


def _train_script(
    run_id: str,
    lane: str,
    *,
    num_rollout: int,
    load_step: int | None,
    exit_after_rollout: int | None = None,
) -> str:
    root = RUN_ROOT / run_id
    model = (root / "model-path.txt").read_text().strip()
    lane_root = root / lane
    save_root = lane_root / "miles"
    export_root = lane_root / "hf-{rollout_id}"
    load = ""
    if load_step is not None:
        load = f"--load {root}/{INTERRUPTED_LANE}/miles --ckpt-step {load_step}"
    bounded_exit = (
        f"--debug-exit-after-rollout {exit_after_rollout}"
        if exit_after_rollout is not None
        else ""
    )
    return textwrap.dedent(
        f"""
        set -euo pipefail
        ray stop --force >/dev/null 2>&1 || true
        pkill -9 sglang >/dev/null 2>&1 || true
        mkdir -p {lane_root}
        source /opt/miles/scripts/models/qwen2.5-0.5B.sh
        ray start --head --node-ip-address 127.0.0.1 --num-gpus 1 --disable-usage-stats
        PYTHONPATH=/opt/miles:/root/Megatron-LM ray job submit \\
          --address=http://127.0.0.1:8265 \\
          --runtime-env-json='{{"env_vars":{{"PYTHONPATH":"/opt/miles:/root/Megatron-LM","CUDA_DEVICE_MAX_CONNECTIONS":"1","NVTE_ALLOW_NONDETERMINISTIC_ALGO":"0","CUBLAS_WORKSPACE_CONFIG":":4096:8"}}}}' \\
          -- python /opt/miles/train.py \\
          --actor-num-nodes 1 --actor-num-gpus-per-node 1 \\
          --debug-train-only \\
          "${{MODEL_ARGS[@]}}" \\
          --hf-checkpoint {model} --ref-load {root}/base_torch_dist \\
          --save {save_root} --save-interval 1 \\
          --save-hf '{export_root}' {load} \\
          --rollout-function-path miles.rollout.sft_rollout.generate_rollout \\
          --prompt-data {root}/control.parquet --input-key messages \\
          --num-rollout {num_rollout} {bounded_exit} \\
          --rollout-batch-size 1 --global-batch-size 1 \\
          --loss-type sft_loss --calculate-per-token-loss \\
          --disable-compute-advantages-and-returns \\
          --optimizer adam --lr 1e-5 --lr-decay-style constant \\
          --weight-decay 0.0 --adam-beta1 0.9 --adam-beta2 0.95 \\
          --tensor-model-parallel-size 1 --pipeline-model-parallel-size 1 \\
          --context-parallel-size 1 --expert-model-parallel-size 1 \\
          --expert-tensor-parallel-size 1 --micro-batch-size 1 \\
          --attention-dropout 0.0 --hidden-dropout 0.0 \\
          --moe-token-dispatcher-type alltoall \\
          --attention-softmax-in-fp32 --attention-backend flash \\
          --seed 1234 2>&1 | tee {lane_root}/train.log
        ray stop --force >/dev/null 2>&1 || true
        """
    )


@app.function(**_function_kwargs())
def train_control(run_id: str) -> dict[str, object]:
    log = _run_shell(_train_script(run_id, CONTROL_LANE, num_rollout=2, load_step=None))
    volume.commit()
    return {"lane": CONTROL_LANE, "log_tail": log[-4000:]}


@app.function(**_function_kwargs())
def train_interrupted(run_id: str) -> dict[str, object]:
    log = _run_shell(
        _train_script(
            run_id,
            INTERRUPTED_LANE,
            num_rollout=2,
            load_step=None,
            exit_after_rollout=1,
        )
    )
    volume.commit()
    return {"lane": INTERRUPTED_LANE, "log_tail": log[-4000:]}


@app.function(**_function_kwargs())
def train_resumed(run_id: str) -> dict[str, object]:
    volume.reload()
    target = RUN_ROOT / run_id / RESUMED_LANE
    if target.exists():
        shutil.rmtree(target)
    log = _run_shell(_train_script(run_id, RESUMED_LANE, num_rollout=2, load_step=0))
    volume.commit()
    return {"lane": RESUMED_LANE, "log_tail": log[-4000:]}


@app.function(
    image=image,
    volumes={str(MOUNT): volume},
    timeout=600,
)
def inspect_artifacts(run_id: str) -> dict[str, object]:
    volume.reload()
    root = RUN_ROOT / run_id
    inventory = {}
    loss_lines = {}
    for lane in (CONTROL_LANE, INTERRUPTED_LANE, RESUMED_LANE):
        lane_root = root / lane
        inventory[lane] = [
            {"path": str(path.relative_to(lane_root)), "bytes": path.stat().st_size}
            for path in sorted(lane_root.rglob("*"))
            if path.is_file()
        ]
        log = (lane_root / "train.log").read_text(errors="replace")
        loss_lines[lane] = [
            line for line in log.splitlines() if "loss" in line.lower() or "iteration" in line.lower()
        ][-40:]
    return {"run_id": run_id, "inventory": inventory, "loss_lines": loss_lines}


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _bytes_sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _tree_hashes(root: Path) -> dict[str, str]:
    if not root.is_dir():
        raise RuntimeError(f"PHASE1_TREE_ROOT_MISSING: {root}")
    hashes = {
        str(path.relative_to(root)): _file_sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }
    if not hashes:
        raise RuntimeError(f"PHASE1_TREE_EMPTY: {root}")
    return hashes


def _update_semantic_digest(digest: object, value: object) -> None:
    digest.update(type(value).__qualname__.encode())
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().contiguous()
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
    elif isinstance(value, dict):
        for key in sorted(value, key=repr):
            _update_semantic_digest(digest, key)
            _update_semantic_digest(digest, value[key])
    elif isinstance(value, (list, tuple)):
        for item in value:
            _update_semantic_digest(digest, item)
    elif isinstance(value, bytes):
        digest.update(value)
    else:
        digest.update(repr(value).encode())


def _semantic_sha256(value: object) -> str:
    digest = hashlib.sha256()
    _update_semantic_digest(digest, value)
    return digest.hexdigest()


def _tree_sha256(root: Path) -> str:
    return _bytes_sha256(_canonical_tree_hashes(root))


def _canonical_tree_hashes(root: Path) -> bytes:
    return json.dumps(_tree_hashes(root), sort_keys=True, separators=(",", ":")).encode()


def _step_loss(log: str, step: int) -> float:
    pattern = re.compile(rf"step {step}: \{{'train/loss': ([0-9.eE+-]+)")
    matches = pattern.findall(log)
    if not matches:
        raise RuntimeError(f"PHASE1_LOSS_MISSING: step={step}")
    return float(matches[-1])


@app.function(image=image, gpu="H100", volumes={str(MOUNT): volume}, timeout=1800)
def compare_resume(run_id: str) -> dict[str, object]:
    volume.reload()
    root = RUN_ROOT / run_id
    control_checkpoint = root / CONTROL_LANE / "miles" / "iter_0000001"
    resumed_checkpoint = root / RESUMED_LANE / "miles" / "iter_0000001"
    control_hashes = _tree_hashes(control_checkpoint)
    resumed_hashes = _tree_hashes(resumed_checkpoint)
    control_payload_hashes = {
        path: value for path, value in control_hashes.items() if path.endswith(".distcp")
    }
    resumed_payload_hashes = {
        path: value for path, value in resumed_hashes.items() if path.endswith(".distcp")
    }
    control_metadata = FileSystemReader(control_checkpoint).read_metadata()
    resumed_metadata = FileSystemReader(resumed_checkpoint).read_metadata()
    metadata_semantics_exact = (
        control_metadata.state_dict_metadata == resumed_metadata.state_dict_metadata
        and control_metadata.planner_data == resumed_metadata.planner_data
        and control_metadata.storage_data == resumed_metadata.storage_data
    )
    required_state_keys = {
        "iteration",
        "checkpoint_version",
        "num_floating_point_operations_so_far",
        "opt_param_scheduler",
        "optimizer",
    }
    control_common = torch.load(
        control_checkpoint / "common.pt", map_location="cpu", weights_only=False
    )
    resumed_common = torch.load(
        resumed_checkpoint / "common.pt", map_location="cpu", weights_only=False
    )
    missing_common_keys = sorted(
        required_state_keys - (control_common.keys() & resumed_common.keys())
    )
    control_common_state = {
        key: control_common[key] for key in sorted(required_state_keys) if key in control_common
    }
    resumed_common_state = {
        key: resumed_common[key] for key in sorted(required_state_keys) if key in resumed_common
    }
    control_common_sha = _semantic_sha256(control_common_state)
    resumed_common_sha = _semantic_sha256(resumed_common_state)
    dcp_state_keys = sorted(control_metadata.state_dict_metadata)
    control_cursor = root / CONTROL_LANE / "miles" / "rollout" / "global_dataset_state_dict_1.pt"
    resumed_cursor = root / RESUMED_LANE / "miles" / "rollout" / "global_dataset_state_dict_1.pt"
    control_loss = _step_loss((root / CONTROL_LANE / "train.log").read_text(), 1)
    resumed_loss = _step_loss((root / RESUMED_LANE / "train.log").read_text(), 1)
    control_export_hashes = _tree_hashes(root / CONTROL_LANE / "hf-1")
    resumed_export_hashes = _tree_hashes(root / RESUMED_LANE / "hf-1")
    evidence = {
        "checkpoint_files_exact": control_hashes == resumed_hashes,
        "checkpoint_file_count": len(control_hashes),
        "checkpoint_relative_paths_exact": set(control_hashes) == set(resumed_hashes),
        "mismatched_checkpoint_files": sorted(
            path
            for path in set(control_hashes) | set(resumed_hashes)
            if control_hashes.get(path) != resumed_hashes.get(path)
        ),
        "dcp_payload_files_exact": control_payload_hashes == resumed_payload_hashes,
        "dcp_metadata_semantics_exact": metadata_semantics_exact,
        "dcp_state_key_count": len(dcp_state_keys),
        "dcp_contains_model_state": any(
            key.startswith(("decoder.", "embedding.")) for key in dcp_state_keys
        ),
        "dcp_contains_optimizer_state": any(
            key.startswith("optimizer.") for key in dcp_state_keys
        ),
        "dcp_contains_rng_state": any(key.startswith("rng_state/") for key in dcp_state_keys),
        "common_training_state_exact": (
            not missing_common_keys and control_common_sha == resumed_common_sha
        ),
        "common_training_state_keys": sorted(control_common_state),
        "common_serialization_fields_excluded": sorted(
            (control_common.keys() | resumed_common.keys()) - required_state_keys
        ),
        "missing_common_training_state_keys": missing_common_keys,
        "control_common_training_state_sha256": control_common_sha,
        "resumed_common_training_state_sha256": resumed_common_sha,
        "data_cursor_exact": _file_sha256(control_cursor) == _file_sha256(resumed_cursor),
        "control_data_cursor_sha256": _file_sha256(control_cursor),
        "resumed_data_cursor_sha256": _file_sha256(resumed_cursor),
        "next_update_loss_exact": control_loss == resumed_loss,
        "control_next_update_loss": control_loss,
        "resumed_next_update_loss": resumed_loss,
        "hf_export_tree_exact": control_export_hashes == resumed_export_hashes,
        "hf_export_file_count": len(control_export_hashes),
        "mismatched_hf_export_files": sorted(
            path
            for path in set(control_export_hashes) | set(resumed_export_hashes)
            if control_export_hashes.get(path) != resumed_export_hashes.get(path)
        ),
    }
    evidence["passed"] = all(
        evidence[key]
        for key in (
            "checkpoint_relative_paths_exact",
            "dcp_payload_files_exact",
            "dcp_metadata_semantics_exact",
            "dcp_contains_model_state",
            "dcp_contains_optimizer_state",
            "dcp_contains_rng_state",
            "common_training_state_exact",
            "data_cursor_exact",
            "next_update_loss_exact",
            "hf_export_tree_exact",
        )
    )
    (root / "resume-parity.json").write_text(json.dumps(evidence, indent=2, sort_keys=True))
    volume.commit()
    if not evidence["passed"]:
        raise RuntimeError(f"PHASE1_MILES_RESUME_PARITY_FAILED: {evidence}")
    return evidence


@app.function(image=image, gpu="H100", volumes={str(MOUNT): volume}, timeout=600)
def inspect_cursors(run_id: str) -> dict[str, object]:
    volume.reload()
    root = RUN_ROOT / run_id
    states = {}
    logs = {}
    common = {}
    for lane in (CONTROL_LANE, INTERRUPTED_LANE, RESUMED_LANE):
        rollout = root / lane / "miles" / "rollout"
        states[lane] = {
            path.name: torch.load(path, map_location="cpu", weights_only=False)
            for path in sorted(rollout.glob("*.pt"))
        }
        logs[lane] = [
            line
            for line in (root / lane / "train.log").read_text().splitlines()
            if "dataset" in line.lower() or "sample_offset" in line.lower()
        ]
        common_path = root / lane / "miles" / "iter_0000001" / "common.pt"
        if common_path.is_file():
            payload = torch.load(common_path, map_location="cpu", weights_only=False)
            common[lane] = {
                key: type(value).__name__
                for key, value in payload.items()
            }
    return {"states": states, "logs": logs, "common": common}


@app.function(
    image=parity_image,
    gpu="H100",
    secrets=[secret],
    volumes={str(MOUNT): volume, str(HF_MOUNT): hf_volume},
    timeout=7200,
)
def sglang_top_k_parity(
    run_id: str, parity_dtype: str = PARITY_DTYPE, full_vocabulary: bool = True
) -> dict[str, object]:
    volume.reload()
    runtime_versions = {
        package: importlib.metadata.version(package)
        for package in ("sglang", "sglang-kernel", "tilelang", "torch")
    }
    expected_versions = {"sglang-kernel": "0.4.6.post1", "tilelang": "0.1.11"}
    if any(runtime_versions[name] != version for name, version in expected_versions.items()):
        raise RuntimeError(
            "PHASE1_SGLANG_RUNTIME_PREFLIGHT_FAILED: "
            f"expected={expected_versions} observed={runtime_versions}"
        )
    export = RUN_ROOT / run_id / RESUMED_LANE / "hf-1"
    if not (export / ".complete").is_file():
        raise RuntimeError("PHASE1_EXPORT_INCOMPLETE")
    tokenizer = AutoTokenizer.from_pretrained(export)
    prompt = "The next integer after 41 is"
    token_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()
    torch_dtype = torch.float32 if parity_dtype == "float32" else torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(export, dtype=torch_dtype).cuda()
    with torch.no_grad():
        training_logprobs = torch.log_softmax(model(token_ids).logits[0, -1].float(), dim=-1)
    vocabulary_size = int(model.config.vocab_size)
    requested_k = vocabulary_size if full_vocabulary else 128
    values, ids = torch.topk(training_logprobs, k=requested_k)
    expected_ids = ids.cpu().tolist()
    expected_values = values.cpu().tolist()
    del model
    torch.cuda.empty_cache()

    env = {
        **os.environ,
        "PYTHONPATH": "/opt/sglang/python",
        "HF_HOME": str(HF_MOUNT),
        "SGLANG_FORCE_FUSED_OP_BACKEND": "torch",
    }
    server = subprocess.Popen(
        [
            "python",
            "-m",
            "sglang.launch_server",
            "--model-path",
            str(export),
            "--host",
            "127.0.0.1",
            "--port",
            "8000",
            "--tp",
            "1",
            "--dtype",
            parity_dtype,
            "--attention-backend",
            "torch_native",
            "--disable-cuda-graph",
            "--mem-fraction-static",
            "0.70",
            "--random-seed",
            "0",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )
    try:
        deadline = time.monotonic() + 600
        while time.monotonic() < deadline:
            try:
                if requests.get("http://127.0.0.1:8000/health", timeout=2).ok:
                    break
            except requests.RequestException:
                pass
            if server.poll() is not None:
                output = server.stdout.read() if server.stdout else ""
                raise RuntimeError(f"PHASE1_SGLANG_START_FAILED:\n{output[-30000:]}")
            time.sleep(2)
        else:
            raise RuntimeError("PHASE1_SGLANG_START_TIMEOUT")
        response = requests.post(
            "http://127.0.0.1:8000/generate",
            json={
                "input_ids": token_ids.cpu().tolist()[0],
                "sampling_params": {
                    "temperature": 0.0,
                    "max_new_tokens": 1,
                    "repetition_penalty": 1.0,
                    "top_k": -1,
                    "top_p": 1.0,
                },
                "return_logprob": True,
                "top_logprobs_num": requested_k,
            },
            timeout=120,
        )
        response.raise_for_status()
        observed = response.json()["meta_info"]["output_top_logprobs"][0]
        observed_values = [float(item[0]) for item in observed]
        observed_ids = [int(item[1]) for item in observed]
        observed_by_id = dict(zip(observed_ids, observed_values, strict=True))
        vocabulary_covered = set(observed_by_id) == set(range(vocabulary_size))
        max_abs_error = max(
            abs(expected - observed_by_id[token_id])
            for expected, token_id in zip(expected_values, expected_ids, strict=True)
            if token_id in observed_by_id
        )
        top_128_ids_exact = observed_ids[:128] == expected_ids[:128]
        tolerance = 0.02
        expected_by_id = dict(zip(expected_ids, expected_values, strict=True))
        rank_diagnostics = {}
        for rank in (1, 8, 16, 32):
            expected_rank_ids = expected_ids[:rank]
            observed_rank_ids = observed_ids[:rank]
            aligned_errors = [
                abs(expected_by_id[token_id] - observed_by_id[token_id])
                for token_id in expected_rank_ids
                if token_id in observed_by_id
            ]
            rank_diagnostics[str(rank)] = {
                "overlap": len(set(expected_rank_ids) & set(observed_rank_ids)),
                "ordered_ids_exact": expected_rank_ids == observed_rank_ids,
                "max_absolute_logprob_error": max(aligned_errors),
                "mean_absolute_logprob_error": sum(aligned_errors) / len(aligned_errors),
            }
        top_1_margin = expected_values[0] - expected_values[1]
        top_1_error = abs(expected_values[0] - observed_by_id[expected_ids[0]])
        passed = (
            (vocabulary_covered if full_vocabulary else True)
            and top_128_ids_exact
            and max_abs_error <= tolerance
        )
        evidence = {
            "mode": "full_vocabulary_logprob" if full_vocabulary else "top_k_logprob",
            "vocabulary_size": vocabulary_size,
            "requested_k": requested_k,
            "absolute_tolerance": tolerance,
            "vocabulary_token_ids_exact": vocabulary_covered,
            "top_128_ordered_token_ids_exact": top_128_ids_exact,
            "max_absolute_logprob_error": max_abs_error,
            "rank_diagnostics": rank_diagnostics,
            "top_1_margin": top_1_margin,
            "top_1_absolute_logprob_error": top_1_error,
            "top_1_margin_to_error_ratio": top_1_margin / top_1_error,
            "passed": passed,
            "miles_revision": MILES_REVISION,
            "sglang_revision": SGLANG_REVISION,
            "sglang_runtime_image": SGLANG_RUNTIME_IMAGE,
            "runtime_versions": runtime_versions,
            "attention_backend": "torch_native",
            "fused_op_backend": "torch",
            "parity_dtype": parity_dtype,
            "prompt": prompt,
            "prompt_token_ids": token_ids.cpu().tolist()[0],
            "expected_top_128_token_ids": expected_ids[:128],
            "observed_top_128_token_ids": observed_ids[:128],
        }
        filename = (
            "sglang-full-vocabulary-parity.json"
            if full_vocabulary
            else f"sglang-{parity_dtype}-diagnostics.json"
        )
        (RUN_ROOT / run_id / filename).write_text(
            json.dumps(evidence, indent=2, sort_keys=True)
        )
        volume.commit()
        if not passed:
            raise RuntimeError(f"PHASE1_SGLANG_TOP_K_PARITY_FAILED: {evidence}")
        return evidence
    finally:
        server.terminate()
        try:
            server.wait(timeout=30)
        except subprocess.TimeoutExpired:
            server.kill()


@app.function(
    image=image,
    gpu="H100",
    volumes={str(MOUNT): volume, str(HF_MOUNT): hf_volume},
    timeout=7200,
)
def publish_accepted_checkpoint(run_id: str) -> dict[str, object]:
    volume.reload()
    root = RUN_ROOT / run_id
    resume_evidence_path = root / "resume-parity.json"
    parity_evidence_path = root / "sglang-full-vocabulary-parity.json"
    bf16_diagnostic_path = root / "sglang-bfloat16-diagnostics.json"
    resume_evidence = json.loads(resume_evidence_path.read_text())
    parity_evidence = json.loads(parity_evidence_path.read_text())
    bf16_diagnostic = json.loads(bf16_diagnostic_path.read_text())
    if not resume_evidence.get("passed"):
        raise RuntimeError("PHASE1_PUBLISH_RESUME_EVIDENCE_REJECTED")
    if not (
        parity_evidence.get("passed")
        and parity_evidence.get("mode") == "full_vocabulary_logprob"
        and parity_evidence.get("parity_dtype") == "float32"
        and parity_evidence.get("vocabulary_token_ids_exact")
    ):
        raise RuntimeError("PHASE1_PUBLISH_INFERENCE_EVIDENCE_REJECTED")
    if bf16_diagnostic.get("passed"):
        raise RuntimeError("PHASE1_PUBLISH_BF16_DIAGNOSTIC_UNEXPECTED_PASS")

    accepted_source = root / "accepted-source"
    if accepted_source.exists():
        shutil.rmtree(accepted_source)
    accepted_source.mkdir()
    shutil.copytree(root / RESUMED_LANE / "hf-1", accepted_source / "model-weights")
    shutil.copytree(root / RESUMED_LANE / "hf-1", accepted_source / "inference-export")
    shutil.copytree(
        root / RESUMED_LANE / "miles" / "iter_0000001",
        accepted_source / "optimizer-scheduler",
    )
    runtime_state = accepted_source / "runtime-state"
    runtime_state.mkdir()
    shutil.copy2(
        root / RESUMED_LANE / "miles" / "rollout" / "global_dataset_state_dict_1.pt",
        runtime_state / "data-cursor.pt",
    )
    shutil.copy2(resume_evidence_path, runtime_state / "resume-parity.json")
    shutil.copy2(parity_evidence_path, runtime_state / "inference-parity.json")
    shutil.copy2(bf16_diagnostic_path, runtime_state / "bf16-diagnostic.json")

    model_path = Path((root / "model-path.txt").read_text().strip())
    base_model_hash = _tree_sha256(model_path)
    export = accepted_source / "inference-export"
    tokenizer_files = {
        path.name: _file_sha256(path)
        for path in export.iterdir()
        if path.is_file()
        and path.name
        in {
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "vocab.json",
            "merges.txt",
        }
    }
    if not tokenizer_files:
        raise RuntimeError("PHASE1_PUBLISH_TOKENIZER_ARTIFACTS_MISSING")
    tokenizer_hash = _bytes_sha256(
        json.dumps(tokenizer_files, sort_keys=True, separators=(",", ":")).encode()
    )
    renderer_hash = _bytes_sha256(
        json.dumps(
            {
                "tokenizer_config_sha256": tokenizer_files.get("tokenizer_config.json"),
                "prompt_transport": "raw_token_ids",
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    )
    toolchain_hash = _bytes_sha256(
        json.dumps(
            {
                "miles": MILES_REVISION,
                "sglang": SGLANG_REVISION,
                "miles_image": MILES_IMAGE,
                "sglang_image": SGLANG_RUNTIME_IMAGE,
                "runtime_versions": parity_evidence["runtime_versions"],
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    )
    native_checkpoint = root / RESUMED_LANE / "miles" / "iter_0000001"
    native_dcp_payload_hashes = {
        path.name: _file_sha256(path)
        for path in native_checkpoint.glob("*.distcp")
    }
    if not native_dcp_payload_hashes or not resume_evidence["dcp_contains_rng_state"]:
        raise RuntimeError("PHASE1_PUBLISH_RNG_STATE_MISSING")
    rng_hash = _bytes_sha256(
        json.dumps(
            native_dcp_payload_hashes, sort_keys=True, separators=(",", ":")
        ).encode()
    )
    draft = CheckpointDraft(
        root=accepted_source,
        identity=CheckpointIdentity(
            checkpoint_id=f"phase1-{run_id}",
            model_arm=ModelArm.CONTROL_QWEN25_05B,
            training_method=TrainingMethod.SFT,
            training_backend="miles",
            global_update=2,
            policy_version=2,
            base_model_id=MODEL_ID,
            base_model_revision=MODEL_REVISION,
            base_model_sha256=base_model_hash,
        ),
        cursor=RuntimeCursor(data=2, rollout=2),
        experiment_sha256=_bytes_sha256(b"phase1-live-miles-sglang-control-v1"),
        config_sha256=_file_sha256(Path(__file__)),
        data_sha256=_file_sha256(root / "control.parquet"),
        rng_sha256=rng_hash,
        runtime=RuntimeBinding(
            python_version=platform.python_version(),
            training_backend_revision=MILES_REVISION,
            inference_backend_revision=SGLANG_REVISION,
            tokenizer_sha256=tokenizer_hash,
            renderer_sha256=renderer_hash,
            toolchain_sha256=toolchain_hash,
            precision="float32-parity/bfloat16-training",
            quantization="none",
        ),
        artifacts={
            ArtifactRole.MODEL_WEIGHTS: Path("model-weights"),
            ArtifactRole.OPTIMIZER_SCHEDULER: Path("optimizer-scheduler"),
            ArtifactRole.RUNTIME_STATE: Path("runtime-state"),
            ArtifactRole.INFERENCE_EXPORT: Path("inference-export"),
        },
    )

    resume_hash = _file_sha256(resume_evidence_path)
    parity_hash = _file_sha256(parity_evidence_path)
    bf16_hash = _file_sha256(bf16_diagnostic_path)

    def validate(package: Path) -> CheckpointValidation:
        packaged_runtime = package / "runtime-state"
        packaged_resume = json.loads((packaged_runtime / "resume-parity.json").read_text())
        packaged_parity = json.loads((packaged_runtime / "inference-parity.json").read_text())
        packaged_bf16 = json.loads((packaged_runtime / "bf16-diagnostic.json").read_text())
        hashes_exact = (
            _file_sha256(packaged_runtime / "resume-parity.json") == resume_hash
            and _file_sha256(packaged_runtime / "inference-parity.json") == parity_hash
            and _file_sha256(packaged_runtime / "bf16-diagnostic.json") == bf16_hash
        )
        return CheckpointValidation(
            resume_exact=bool(hashes_exact and packaged_resume.get("passed")),
            inference_parity_passed=bool(
                hashes_exact
                and packaged_parity.get("passed")
                and packaged_parity.get("parity_dtype") == "float32"
                and packaged_parity.get("vocabulary_token_ids_exact")
                and not packaged_bf16.get("passed")
            ),
            inference_parity_mode="full_vocabulary_logprob",
            details={
                "resume_evidence_sha256": resume_hash,
                "inference_evidence_sha256": parity_hash,
                "bf16_diagnostic_sha256": bf16_hash,
                "bf16_is_non_acceptance_diagnostic": True,
                "native_dcp_payload_sha256": native_dcp_payload_hashes,
                "cursor_is_next_data_and_rollout": True,
            },
        )

    checkpoint = FilesystemCheckpointStore(ACCEPTED_STORE_ROOT).publish(draft, validate)
    volume.commit()
    return {
        "checkpoint_id": checkpoint.identity.checkpoint_id,
        "manifest_sha256": checkpoint.manifest_sha256,
        "validation_sha256": checkpoint.validation_sha256,
        "acceptance_sha256": checkpoint.acceptance_sha256,
        "latest": str(ACCEPTED_STORE_ROOT / "latest.json"),
        "resume_evidence_sha256": resume_hash,
        "inference_evidence_sha256": parity_hash,
        "bf16_diagnostic_sha256": bf16_hash,
    }


@app.function(image=image, volumes={str(MOUNT): volume}, timeout=1800)
def verify_accepted_checkpoint() -> dict[str, object]:
    volume.reload()
    checkpoint = FilesystemCheckpointStore(ACCEPTED_STORE_ROOT).latest()
    expected_checkpoint_id = "phase1-qwen25-control-b12e6e2990a04103ac8c51a7d3ad4267"
    expected_manifest = "6f9945363a033bad8b8113bf9b0a5cb50b666c45a593b993565684183c061dff"
    expected_resume = "2c064eac1cd70f15fc38e4e2842b52e3cc16c9e16ce86d2786437bf0678632bc"
    expected_inference = "685c1db594d18a94a1bc9109cc7c743d6d1d57e488967e9b2c98ccbbc6bd2ae1"
    if checkpoint.identity.checkpoint_id != expected_checkpoint_id:
        raise RuntimeError("PHASE1_ACCEPTED_CHECKPOINT_ID_MISMATCH")
    if checkpoint.manifest_sha256 != expected_manifest:
        raise RuntimeError("PHASE1_ACCEPTED_MANIFEST_MISMATCH")
    details = checkpoint.validation.details
    if details.get("resume_evidence_sha256") != expected_resume:
        raise RuntimeError("PHASE1_ACCEPTED_RESUME_EVIDENCE_MISMATCH")
    if details.get("inference_evidence_sha256") != expected_inference:
        raise RuntimeError("PHASE1_ACCEPTED_INFERENCE_EVIDENCE_MISMATCH")
    payload_files = _tree_hashes(checkpoint.path / "payload")
    return {
        "checkpoint_id": checkpoint.identity.checkpoint_id,
        "manifest_sha256": checkpoint.manifest_sha256,
        "validation_sha256": checkpoint.validation_sha256,
        "acceptance_sha256": checkpoint.acceptance_sha256,
        "resume_evidence_sha256": details["resume_evidence_sha256"],
        "inference_evidence_sha256": details["inference_evidence_sha256"],
        "payload_file_count": len(payload_files),
        "payload_sha256": _bytes_sha256(
            json.dumps(payload_files, sort_keys=True, separators=(",", ":")).encode()
        ),
        "passed": True,
    }


@app.local_entrypoint()
def main() -> None:
    run_id = f"qwen25-control-{uuid.uuid4().hex}"
    stages = {
        "preflight": preflight.remote(),
        "prepare": prepare_control.remote(run_id),
        "control": train_control.remote(run_id),
        "interrupted": train_interrupted.remote(run_id),
        "resumed": train_resumed.remote(run_id),
    }
    print(json.dumps({"run_id": run_id, "stages": stages}, indent=2, sort_keys=True))


@app.local_entrypoint()
def inspect_main(run_id: str) -> None:
    print(json.dumps(inspect_artifacts.remote(run_id), indent=2, sort_keys=True))


@app.local_entrypoint()
def parity_main(run_id: str) -> None:
    print(json.dumps(sglang_top_k_parity.remote(run_id), indent=2, sort_keys=True))


@app.local_entrypoint()
def bf16_diagnostics_main(run_id: str) -> None:
    print(
        json.dumps(
            sglang_top_k_parity.remote(run_id, "bfloat16", False),
            indent=2,
            sort_keys=True,
        )
    )


@app.local_entrypoint()
def publish_main(run_id: str) -> None:
    print(json.dumps(publish_accepted_checkpoint.remote(run_id), indent=2, sort_keys=True))


@app.local_entrypoint()
def verify_accepted_main() -> None:
    print(json.dumps(verify_accepted_checkpoint.remote(), indent=2, sort_keys=True))


@app.local_entrypoint()
def compare_main(run_id: str) -> None:
    print(json.dumps(compare_resume.remote(run_id), indent=2, sort_keys=True))


@app.local_entrypoint()
def resume_main(run_id: str) -> None:
    print(json.dumps(train_resumed.remote(run_id), indent=2, sort_keys=True))


@app.local_entrypoint()
def cursor_main(run_id: str) -> None:
    print(json.dumps(inspect_cursors.remote(run_id), indent=2, sort_keys=True))
