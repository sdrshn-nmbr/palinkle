"""Prepare the pinned Laguna DSpark checkpoint and launch vLLM."""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from huggingface_hub import snapshot_download
import modal

from opjax.pallas.laguna_speculative import (
    DSPARK_ID,
    DSPARK_REVISION,
    VLLM_IMAGE,
    VLLM_OBSERVED_BUILD,
    canonical_sha256,
    normalize_dspark_config,
)


def _prepare_dspark_snapshot(
    *, root: Path, model: str = DSPARK_ID, revision: str | None = DSPARK_REVISION
) -> Path:
    model_path = Path(model)
    source = (
        model_path
        if model_path.is_dir()
        else Path(snapshot_download(repo_id=model, revision=revision))
    )
    identity = revision or canonical_sha256(
        {"model": str(source.resolve()), "config": (source / "config.json").read_text()}
    )
    destination = root / identity
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
        snapshot = _prepare_dspark_snapshot(
            root=Path("/tmp/opjax-dspark"),
            model=str(config["model"]),
            revision=config.get("revision"),
        )
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


def _start_artifact_commits() -> None:
    volume_name = os.environ["OPJAX_SPEC_ARTIFACT_VOLUME"]
    environment_name = os.environ["OPJAX_SPEC_MODAL_ENVIRONMENT"]
    volume = modal.Volume.from_name(
        volume_name,
        environment_name=environment_name,
        version=1,
    )

    def commit_forever() -> None:
        while True:
            time.sleep(60)
            try:
                volume.commit()
            except Exception as exc:
                print(
                    f"LAGUNA_ARTIFACT_COMMIT_FAILED:{type(exc).__name__}:{exc}",
                    file=sys.stderr,
                    flush=True,
                )

    threading.Thread(target=commit_forever, daemon=True).start()


def _write_runtime_fingerprint(*, artifact_dir: Path, arm: str) -> None:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    version_result = subprocess.run(
        ["vllm", "--version"],
        check=True,
        capture_output=True,
        text=True,
    )
    observed_build = version_result.stdout.strip()
    if observed_build != VLLM_OBSERVED_BUILD:
        raise RuntimeError(
            "LAGUNA_VLLM_BUILD_MISMATCH:"
            f"expected={VLLM_OBSERVED_BUILD}:observed={observed_build}"
        )
    fingerprint: dict[str, Any] = {
        "schema_version": 1,
        "arm": arm,
        "image": VLLM_IMAGE,
        "vllm_observed_build": observed_build,
        "vllm_source_revision": "unavailable_in_image_build_metadata",
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
    _start_artifact_commits()
    os.execvp("vllm", ["vllm", "serve", *arguments])


if __name__ == "__main__":
    main()
