"""Modal runner for forced-prefix Laguna DSpark multi-round conformance."""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import uuid

from huggingface_hub import snapshot_download
from huggingface_hub import HfApi
import modal
import numpy as np

from opjax.pallas.laguna_dspark_multiround import (
    build_contexts,
    build_final_report,
    build_multiround_report,
    validate_final_report,
    validate_multiround_report,
)
from opjax.pallas.laguna_speculative import (
    BASH_TOOL,
    TARGET_ID,
    TARGET_REVISION,
    canonical_sha256,
)
from opjax.remote.config import (
    HF_CACHE_DIR,
    HF_CACHE_VOLUME_NAME,
    MODAL_ENVIRONMENT,
    MODAL_SECRET_NAME,
    MODAL_VOLUME_VERSION,
    REMOTE_ENV,
)


APP_NAME = "opjax-laguna-multiround-conformance-v1"
ROOT = Path("/mnt/training")
ARTIFACT_ROOT = Path("/mnt/multiround")
DEEPSPEC_REVISION = "787db11ea347ac3944233e5aa9c7f1bd8a9b5ced"

app = modal.App(APP_NAME)
cache = modal.Volume.from_name(
    HF_CACHE_VOLUME_NAME,
    environment_name=MODAL_ENVIRONMENT,
    create_if_missing=True,
    version=MODAL_VOLUME_VERSION,
)


def _prepare_capture_output(
    output_root: Path, *, telemetry_path: Path
) -> dict[str, object] | None:
    summary_path = output_root / "summary.json"
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        expected = summary.get("summary_sha256")
        unsigned = {key: value for key, value in summary.items() if key != "summary_sha256"}
        if expected != canonical_sha256(unsigned):
            raise RuntimeError(f"MULTIROUND_EXISTING_SUMMARY_INVALID:{output_root}")
        return summary
    if not output_root.exists():
        return None
    archive_root = (
        output_root.parents[1]
        / "attempts"
        / output_root.parent.name
        / output_root.name
        / uuid.uuid4().hex
    )
    archive_root.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(output_root), str(archive_root))
    if telemetry_path.is_file():
        shutil.move(str(telemetry_path), str(archive_root / telemetry_path.name))
    files = []
    for path in sorted(item for item in archive_root.rglob("*") if item.is_file()):
        files.append(
            {
                "path": str(path.relative_to(archive_root)),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "bytes": path.stat().st_size,
            }
        )
    manifest = {
        "schema_version": 1,
        "reason": "incomplete_capture_resumed",
        "original_path": str(output_root.relative_to(output_root.parents[1])),
        "files": files,
    }
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    (archive_root / "archive-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return None


def _content_tree(
    root: Path, *, excluded_names: frozenset[str] = frozenset()
) -> dict[str, object]:
    files = []
    total_bytes = 0
    for path in sorted(
        item
        for item in root.rglob("*")
        if item.is_file()
        and item.name not in excluded_names
        and ".cache" not in item.relative_to(root).parts
    ):
        size = path.stat().st_size
        total_bytes += size
        files.append(
            {
                "path": str(path.relative_to(root)),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "bytes": size,
            }
        )
    return {
        "files": len(files),
        "bytes": total_bytes,
        "sha256": canonical_sha256(files),
    }
training = modal.Volume.from_name(
    "opjax-laguna-speculator-training-v1",
    environment_name=MODAL_ENVIRONMENT,
    create_if_missing=True,
    version=MODAL_VOLUME_VERSION,
)
artifacts = modal.Volume.from_name(
    "opjax-laguna-multiround-conformance-v1",
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
            "OPJAX_SPEC_ARTIFACT_VOLUME": "opjax-laguna-multiround-conformance-v1",
            "OPJAX_SPEC_MODAL_ENVIRONMENT": MODAL_ENVIRONMENT,
        }
    )
    .add_local_python_source("opjax")
)
VOLUMES = {HF_CACHE_DIR: cache, str(ROOT): training, str(ARTIFACT_ROOT): artifacts}
OPTIONS = {"volumes": VOLUMES, "secrets": [secret], "timeout": 21_600}


def _target_path() -> Path:
    return Path(
        snapshot_download(
            TARGET_ID,
            revision=TARGET_REVISION,
            local_dir="/tmp/opjax-laguna-target",
        )
    )


def _telemetry(path: Path) -> tuple[subprocess.Popen[bytes], object]:
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


def _normalize_messages(messages: list[dict[str, object]]) -> list[dict[str, object]]:
    normalized: list[dict[str, object]] = []
    for message in messages:
        value = dict(message)
        calls = value.get("tool_calls")
        if isinstance(calls, list):
            normalized_calls = []
            for call in calls:
                if not isinstance(call, dict) or not isinstance(call.get("function"), dict):
                    raise ValueError("MULTIROUND_TOOL_CALL_INVALID")
                normalized_call = dict(call)
                function = dict(call["function"])
                arguments = function.get("arguments")
                if isinstance(arguments, str):
                    function["arguments"] = json.loads(arguments)
                normalized_call["function"] = function
                normalized_calls.append(normalized_call)
            value["tool_calls"] = normalized_calls
        normalized.append(value)
    return normalized


@app.function(image=deepspec_image, **OPTIONS)
def prepare_matrix(
    run_id: str, prompt_id: str, messages: list[dict[str, object]]
) -> dict[str, list[int]]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        _target_path(), trust_remote_code=True, fix_mistral_regex=True
    )
    rendered = tokenizer.apply_chat_template(
        _normalize_messages(messages),
        tools=[BASH_TOOL],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    if isinstance(rendered, Mapping):
        rendered = rendered["input_ids"]
    if hasattr(rendered, "tolist"):
        rendered = rendered.tolist()
    if rendered and isinstance(rendered[0], list):
        rendered = rendered[0]
    contexts = build_contexts([int(token) for token in rendered])
    root = ARTIFACT_ROOT / run_id
    root.mkdir(parents=True, exist_ok=False)
    matrix = {
        "schema_version": 1,
        "prompt_id": prompt_id,
        "rendered_token_count": len(rendered),
        "contexts": contexts,
    }
    (root / "matrix.json").write_text(
        json.dumps(matrix, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    artifacts.commit()
    return contexts


@app.function(image=deepspec_image, **OPTIONS)
def clone_source_run(source_run_id: str, run_id: str) -> dict[str, object]:
    source_root = ARTIFACT_ROOT / source_run_id
    target_root = ARTIFACT_ROOT / run_id
    target_root.mkdir(parents=True, exist_ok=False)
    shutil.copy2(source_root / "matrix.json", target_root / "matrix.json")
    shutil.copytree(source_root / "source", target_root / "source")
    for telemetry in source_root.glob("source-*-gpu.csv"):
        shutil.copy2(telemetry, target_root / telemetry.name)
    sequential_source = source_root / "sequential-source"
    sequential_source_cloned = sequential_source.is_dir()
    if sequential_source_cloned:
        shutil.copytree(sequential_source, target_root / "sequential-source")
        for telemetry in source_root.glob("sequential-source-*-gpu.csv"):
            shutil.copy2(telemetry, target_root / telemetry.name)
    artifacts.commit()
    matrix = json.loads((target_root / "matrix.json").read_text(encoding="utf-8"))
    return {
        "contexts": matrix["contexts"],
        "sequential_source_cloned": sequential_source_cloned,
    }


@app.function(image=deepspec_image, **OPTIONS)
def load_contexts(run_id: str) -> dict[str, list[int]]:
    matrix = json.loads((ARTIFACT_ROOT / run_id / "matrix.json").read_text())
    contexts = matrix.get("contexts")
    if not isinstance(contexts, dict):
        raise RuntimeError("MULTIROUND_CONTEXTS_MISSING")
    return {str(key): [int(token) for token in value] for key, value in contexts.items()}


@app.function(image=deepspec_image, gpu="H200", **OPTIONS)
def capture_source_context(
    run_id: str, context_id: str, base_token_ids: list[int]
) -> list[str]:
    from opjax.remote.laguna_deepspec_conformance import run_capture

    root = ARTIFACT_ROOT / run_id
    target = _target_path()
    draft = ROOT / "selected" / "dspark"
    committed = list(base_token_ids)
    telemetry, output = _telemetry(root / f"source-{context_id}-gpu.csv")
    manifests: list[str] = []
    try:
        for round_index in range(3):
            cell_id = f"{context_id}--round-{round_index}"
            manifest = run_capture(
                output_root=root / "source" / cell_id,
                prompt="",
                target_path=target,
                draft_path=draft,
                input_token_ids=committed,
            )
            anchor_item = manifest["boundaries"]["anchor_token_id"]
            anchor = np.load(
                root / "source" / cell_id / anchor_item["path"], allow_pickle=False
            ).reshape(-1)
            if anchor.size != 1:
                raise RuntimeError(f"MULTIROUND_SOURCE_ANCHOR_INVALID:{cell_id}")
            committed.append(int(anchor[0]))
            manifests.append(str(manifest["manifest_sha256"]))
    finally:
        telemetry.terminate()
        telemetry.wait(timeout=30)
        output.close()
        artifacts.commit()
    return manifests


@app.function(image=vllm_image, gpu="H200", **OPTIONS)
def capture_adapter_context(
    run_id: str,
    context_id: str,
    lane: str,
    expected_processed_starts: list[int] | None = None,
) -> dict[str, object]:
    from opjax.remote.laguna_vllm_conformance import run_token_rounds_capture

    root = ARTIFACT_ROOT / run_id
    telemetry, output = _telemetry(root / f"{lane}-{context_id}-gpu.csv")
    try:
        result = run_token_rounds_capture(
            output_root=root / lane / context_id,
            context_id=context_id,
            source_root=root / "source",
            lane=lane,
            draft_model=str(ROOT / "selected" / "dspark"),
            expected_processed_starts=expected_processed_starts,
        )
    finally:
        telemetry.terminate()
        telemetry.wait(timeout=30)
        output.close()
        artifacts.commit()
    return result


@app.function(image=vllm_image, gpu="H200", **OPTIONS)
def capture_adapter_backend_probe(
    source_run_id: str,
    probe_run_id: str,
    context_id: str,
    attention_backend: str,
    expected_processed_starts: list[int],
    enable_prefix_caching: bool = True,
) -> dict[str, object]:
    """Run one injected forced-prefix context against frozen source evidence."""
    from opjax.remote.laguna_vllm_conformance import run_token_rounds_capture

    source_root = ARTIFACT_ROOT / source_run_id / "source"
    output_root = (
        ARTIFACT_ROOT
        / probe_run_id
        / attention_backend.lower()
        / context_id
    )
    telemetry_path = (
        ARTIFACT_ROOT
        / probe_run_id
        / f"{attention_backend.lower()}-{context_id}-gpu.csv"
    )
    telemetry_path.parent.mkdir(parents=True, exist_ok=True)
    existing = _prepare_capture_output(output_root, telemetry_path=telemetry_path)
    if existing is not None:
        return existing
    telemetry, output = _telemetry(telemetry_path)
    try:
        result = run_token_rounds_capture(
            output_root=output_root,
            context_id=context_id,
            source_root=source_root,
            lane="injected",
            draft_model=str(ROOT / "selected" / "dspark"),
            expected_processed_starts=expected_processed_starts,
            attention_backend=attention_backend,
            enable_prefix_caching=enable_prefix_caching,
        )
    finally:
        telemetry.terminate()
        telemetry.wait(timeout=30)
        output.close()
        artifacts.commit()
    return result


@app.function(image=vllm_image, gpu="H200", **OPTIONS)
def capture_sequential_native_context(
    run_id: str, context_id: str, prompt_token_ids: list[int]
) -> dict[str, object]:
    from opjax.remote.laguna_vllm_conformance import run_sequential_capture

    root = ARTIFACT_ROOT / run_id
    output_root = root / "sequential-native" / context_id
    telemetry_path = root / f"sequential-native-{context_id}-gpu.csv"
    existing = _prepare_capture_output(output_root, telemetry_path=telemetry_path)
    if existing is not None:
        return existing
    telemetry, output = _telemetry(telemetry_path)
    try:
        result = run_sequential_capture(
            output_root=output_root,
            context_id=context_id,
            prompt_token_ids=prompt_token_ids,
            draft_model=str(ROOT / "selected" / "dspark"),
        )
    finally:
        telemetry.terminate()
        telemetry.wait(timeout=30)
        output.close()
        artifacts.commit()
    return result


@app.function(image=deepspec_image, gpu="H200", **OPTIONS)
def capture_sequential_source_context(
    run_id: str, context_id: str
) -> list[str]:
    from opjax.remote.laguna_deepspec_conformance import run_capture

    root = ARTIFACT_ROOT / run_id
    target = _target_path()
    draft = ROOT / "selected" / "dspark"
    telemetry, output = _telemetry(root / f"sequential-source-{context_id}-gpu.csv")
    manifests: list[str] = []
    try:
        for round_index in range(3):
            cell_id = f"{context_id}--round-{round_index}"
            native = json.loads(
                (
                    root
                    / "sequential-native"
                    / context_id
                    / "cells"
                    / cell_id
                    / "manifest.json"
                ).read_text(encoding="utf-8")
            )
            manifest = run_capture(
                output_root=(
                    root / "sequential-source" / context_id / "cells" / cell_id
                ),
                prompt="",
                target_path=target,
                draft_path=draft,
                input_token_ids=native["prompt_token_ids"],
            )
            manifests.append(str(manifest["manifest_sha256"]))
    finally:
        telemetry.terminate()
        telemetry.wait(timeout=30)
        output.close()
        artifacts.commit()
    return manifests


@app.function(image=vllm_image, gpu="H200", **OPTIONS)
def capture_sequential_injected_context(
    run_id: str, context_id: str, prompt_token_ids: list[int]
) -> dict[str, object]:
    from opjax.remote.laguna_vllm_conformance import run_sequential_capture

    root = ARTIFACT_ROOT / run_id
    overrides: list[Path] = []
    for round_index in range(3):
        cell_id = f"{context_id}--round-{round_index}"
        source_root = root / "sequential-source" / context_id / "cells" / cell_id
        source = json.loads((source_root / "manifest.json").read_text(encoding="utf-8"))
        native_root = root / "sequential-native" / context_id / "cells" / cell_id
        native = json.loads((native_root / "manifest.json").read_text(encoding="utf-8"))
        raw = np.load(
            source_root / source["boundaries"]["raw_target_features"]["path"],
            allow_pickle=False,
        )
        start = int(native["processed_token_start"])
        override = raw[:, start:, :]
        override_path = root / "sequential-overrides" / context_id / f"round-{round_index}.npy"
        override_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(override_path, override)
        overrides.append(override_path)
    output_root = root / "sequential-injected" / context_id
    telemetry_path = root / f"sequential-injected-{context_id}-gpu.csv"
    existing = _prepare_capture_output(output_root, telemetry_path=telemetry_path)
    if existing is not None:
        return existing
    telemetry, output = _telemetry(telemetry_path)
    try:
        result = run_sequential_capture(
            output_root=output_root,
            context_id=context_id,
            prompt_token_ids=prompt_token_ids,
            draft_model=str(ROOT / "selected" / "dspark"),
            target_feature_overrides=overrides,
        )
    finally:
        telemetry.terminate()
        telemetry.wait(timeout=30)
        output.close()
        artifacts.commit()
    return result


@app.function(image=deepspec_image, cpu=4, memory=8192, **OPTIONS)
def finalize_forced(run_id: str) -> dict[str, object]:
    root = ARTIFACT_ROOT / run_id
    report = build_multiround_report(root)
    validate_multiround_report(report, root=root)
    (root / "forced-report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    artifacts.commit()
    return report


@app.function(image=deepspec_image, cpu=4, memory=8192, **OPTIONS)
def finalize(run_id: str) -> dict[str, object]:
    root = ARTIFACT_ROOT / run_id
    report = build_final_report(root)
    validate_final_report(report, root=root)
    (root / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    artifacts.commit()
    return report


@app.function(
    image=deepspec_image,
    cpu=4,
    memory=8192,
    volumes=VOLUMES,
    secrets=[secret],
    timeout=21_600,
)
def publish_evidence(
    run_id: str,
    repo_id: str = "sdrshn-nmbr/opjax-laguna-dspark-multiround-v8",
) -> dict[str, object]:
    root = ARTIFACT_ROOT / run_id
    report = json.loads((root / "report.json").read_text(encoding="utf-8"))
    validate_final_report(report, root=root)
    excluded_names = frozenset({"remote-evidence.json"})
    tree = _content_tree(root, excluded_names=excluded_names)
    api = HfApi()
    api.create_repo(repo_id, repo_type="dataset", private=True, exist_ok=True)
    api.upload_large_folder(
        repo_id=repo_id,
        repo_type="dataset",
        folder_path=root,
        ignore_patterns=["**/remote-evidence.json", ".cache/**"],
        num_workers=4,
        print_report_every=60,
    )
    revision = api.repo_info(repo_id, repo_type="dataset").sha
    download_root = Path("/tmp/opjax-multiround-clean-download")
    if download_root.exists():
        shutil.rmtree(download_root)
    snapshot = Path(
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            revision=revision,
            local_dir=download_root,
        )
    )
    clean_root = snapshot
    clean_tree = _content_tree(clean_root, excluded_names=excluded_names)
    if clean_tree != tree:
        raise RuntimeError(f"MULTIROUND_REMOTE_TREE_MISMATCH:{tree}:{clean_tree}")
    clean_report = json.loads(
        (clean_root / "report.json").read_text(encoding="utf-8")
    )
    validate_final_report(clean_report, root=clean_root)
    result = {
        "schema_version": 1,
        "kind": "laguna_dspark_multiround_remote_evidence",
        "run_id": run_id,
        "report_sha256": report["report_sha256"],
        "repo_id": repo_id,
        "repo_type": "dataset",
        "private": True,
        "revision": revision,
        "path_in_repo": ".",
        "content_tree": tree,
        "clean_download_validated": True,
    }
    result["sha256"] = canonical_sha256(result)
    (root / "remote-evidence.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    artifacts.commit()
    return result


@app.local_entrypoint()
def main(
    run_id: str = "",
    corpus: str = "data/pallas/runs/laguna-speculative-v1/replay-corpus.json",
    source_run_id: str = "",
    resume_sequential_only: bool = False,
) -> None:
    payload = json.loads(Path(corpus).read_text(encoding="utf-8"))
    record = max(payload["records"], key=lambda item: item["historical_completion_tokens"])
    resolved_run_id = run_id or f"laguna-multiround-{uuid.uuid4().hex[:12]}"
    if resume_sequential_only:
        if not run_id or source_run_id:
            raise ValueError("MULTIROUND_RESUME_ARGUMENTS_INVALID")
        contexts = load_contexts.remote(resolved_run_id)
        sequential_source_cloned = True
    elif source_run_id:
        clone = clone_source_run.remote(source_run_id, resolved_run_id)
        contexts = clone["contexts"]
        sequential_source_cloned = bool(clone["sequential_source_cloned"])
    else:
        contexts = prepare_matrix.remote(
            resolved_run_id, record["prompt_id"], record["messages"]
        )
        source_calls = [
            capture_source_context.spawn(resolved_run_id, context_id, tokens)
            for context_id, tokens in sorted(contexts.items())
        ]
        for call in source_calls:
            call.get()
        sequential_source_cloned = False
    if not resume_sequential_only:
        native_calls = {
            context_id: capture_adapter_context.spawn(
                resolved_run_id, context_id, "native"
            )
            for context_id in sorted(contexts)
        }
        native_results = {
            context_id: call.get() for context_id, call in native_calls.items()
        }
        injected_calls = [
            capture_adapter_context.spawn(
                resolved_run_id,
                context_id,
                "injected",
                [
                    int(value)
                    for value in native_results[context_id]["processed_token_starts"]
                ],
            )
            for context_id in sorted(contexts)
        ]
        for call in injected_calls:
            call.get()
    sequential_native_calls = [
        capture_sequential_native_context.spawn(
            resolved_run_id, context_id, contexts[context_id]
        )
        for context_id in sorted(contexts)
    ]
    for call in sequential_native_calls:
        call.get()
    if not sequential_source_cloned:
        sequential_source_calls = [
            capture_sequential_source_context.spawn(resolved_run_id, context_id)
            for context_id in sorted(contexts)
        ]
        for call in sequential_source_calls:
            call.get()
    sequential_injected_calls = [
        capture_sequential_injected_context.spawn(
            resolved_run_id, context_id, contexts[context_id]
        )
        for context_id in sorted(contexts)
    ]
    for call in sequential_injected_calls:
        call.get()
    print(json.dumps(finalize.remote(resolved_run_id), indent=2, sort_keys=True))
