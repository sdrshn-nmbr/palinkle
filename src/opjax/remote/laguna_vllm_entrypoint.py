"""Prepare the pinned Laguna DSpark checkpoint and launch vLLM."""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any

from huggingface_hub import snapshot_download

from opjax.pallas.laguna_speculative import (
    DSPARK_ID,
    DSPARK_REVISION,
    VLLM_IMAGE,
    VLLM_SOURCE_REVISION,
    canonical_sha256,
    normalize_dspark_config,
)


def _prepare_dspark_snapshot(*, root: Path) -> Path:
    source = Path(
        snapshot_download(
            repo_id=DSPARK_ID,
            revision=DSPARK_REVISION,
        )
    )
    destination = root / DSPARK_REVISION
    destination.mkdir(parents=True, exist_ok=True)
    for source_path in source.iterdir():
        if source_path.name == "config.json":
            continue
        target = destination / source_path.name
        if target.exists() or target.is_symlink():
            target.unlink()
        target.symlink_to(source_path)
    config = normalize_dspark_config(
        json.loads((source / "config.json").read_text(encoding="utf-8"))
    )
    (destination / "config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return destination


def _rewrite_speculative_config(arguments: list[str]) -> tuple[list[str], str]:
    rewritten = list(arguments)
    if "--speculative-config" not in rewritten:
        return rewritten, "plain"
    index = rewritten.index("--speculative-config") + 1
    config: dict[str, Any] = json.loads(rewritten[index])
    method = str(config.get("method"))
    if method == "dspark":
        snapshot = _prepare_dspark_snapshot(root=Path("/tmp/opjax-dspark"))
        config["model"] = str(snapshot)
        config.pop("revision", None)
    rewritten[index] = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return rewritten, method


def _start_gpu_telemetry(*, artifact_dir: Path) -> None:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    output = (artifact_dir / "gpu.csv").open("ab")
    command = [
        "nvidia-smi",
        "--query-gpu=timestamp,index,name,memory.used,memory.total,utilization.gpu,power.draw",
        "--format=csv",
        "--loop=1",
    ]
    subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT)


def _write_runtime_fingerprint(*, artifact_dir: Path, arm: str) -> None:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    fingerprint: dict[str, Any] = {
        "schema_version": 1,
        "arm": arm,
        "image": VLLM_IMAGE,
        "vllm_source_revision": VLLM_SOURCE_REVISION,
        "python": sys.version,
        "platform": platform.platform(),
        "argv": sys.argv,
    }
    fingerprint["sha256"] = canonical_sha256(fingerprint)
    (artifact_dir / "runtime.json").write_text(
        json.dumps(fingerprint, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    arguments, arm = _rewrite_speculative_config(sys.argv[1:])
    artifact_root = Path(os.environ.get("OPJAX_SPEC_ARTIFACT_ROOT", "/tmp/opjax-spec"))
    run_id = os.environ.get("OPJAX_SPEC_RUN_ID") or uuid.uuid4().hex
    artifact_dir = artifact_root / arm / run_id
    _write_runtime_fingerprint(artifact_dir=artifact_dir, arm=arm)
    _start_gpu_telemetry(artifact_dir=artifact_dir)
    os.execvp("vllm", ["vllm", "serve", *arguments])


if __name__ == "__main__":
    main()
