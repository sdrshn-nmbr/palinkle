"""Torch-free observed GPU runtime identity."""

from __future__ import annotations

from collections.abc import Callable
import hashlib
import json
import subprocess
from typing import Any


CommandRunner = Callable[..., subprocess.CompletedProcess[str]]


def _canonical_sha256(value: dict[str, Any]) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def gpu_runtime_identity(
    *, run: CommandRunner = subprocess.run, expected_count: int = 1
) -> dict[str, Any]:
    command = (
        "nvidia-smi",
        "--query-gpu=name,uuid,driver_version,memory.total,compute_cap",
        "--format=csv,noheader,nounits",
    )
    try:
        result = run(
            command,
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
    except FileNotFoundError as error:
        raise RuntimeError("GPU_TOOL_MISSING:nvidia-smi") from error
    if result.returncode != 0:
        raise RuntimeError(
            f"GPU_NVIDIA_SMI_FAILED:{result.returncode}:{result.stderr.strip()}"
        )
    rows = []
    for line in result.stdout.splitlines():
        values = [value.strip() for value in line.split(",")]
        if len(values) != 5 or any(not value for value in values):
            raise RuntimeError(f"GPU_NVIDIA_SMI_ROW_INVALID:{line}")
        rows.append(
            {
                "name": values[0],
                "uuid": values[1],
                "driver_version": values[2],
                "memory_total_mib": int(values[3]),
                "compute_capability": values[4],
            }
        )
    if expected_count < 1 or len(rows) != expected_count:
        raise RuntimeError(
            f"GPU_DEVICE_COUNT_INVALID:{len(rows)}:expected={expected_count}"
        )
    identity = {"schema_version": 1, "devices": rows}
    identity["sha256"] = _canonical_sha256(identity)
    return identity
