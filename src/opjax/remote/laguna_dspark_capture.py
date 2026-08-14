"""Opt-in tensor capture for Laguna DSpark differential conformance."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import threading
from typing import Any

import numpy as np
import torch


_LOCK = threading.Lock()
_COUNTERS: dict[tuple[str, str], int] = {}


def _control() -> tuple[Path, str] | None:
    root_value = os.environ.get("OPJAX_DSPARK_CAPTURE_ROOT")
    if not root_value:
        return None
    root = Path(root_value)
    control = root / "active.json"
    if not control.is_file():
        return None
    value = json.loads(control.read_text(encoding="utf-8"))
    session = value.get("session")
    if not isinstance(session, str) or not session:
        raise RuntimeError("DSPARK_CAPTURE_SESSION_INVALID")
    return root, session


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_tensor(*, root: Path, session: str, name: str, value: torch.Tensor) -> None:
    with _LOCK:
        key = (session, name)
        index = _COUNTERS.get(key, 0)
        _COUNTERS[key] = index + 1
        session_root = root / session
        session_root.mkdir(parents=True, exist_ok=True)
        path = session_root / f"{name}-{index:03d}.npy"
        temporary = path.with_suffix(".tmp")
        tensor = value.detach().contiguous().cpu()
        if tensor.dtype == torch.bfloat16:
            tensor = tensor.float()
        array = tensor.numpy()
        with temporary.open("wb") as handle:
            np.save(handle, array, allow_pickle=False)
        os.replace(temporary, path)
        record: dict[str, Any] = {
            "name": name,
            "index": index,
            "path": path.name,
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            "source_dtype": str(value.dtype),
            "sha256": _sha256(path),
        }
        ledger = session_root / "ledger.jsonl"
        with ledger.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def capture_tensor(name: str, value: torch.Tensor) -> None:
    active = _control()
    if active is None:
        return
    root, session = active
    _write_tensor(root=root, session=session, name=name, value=value)


def capture_is_active() -> bool:
    return _control() is not None


def capture_is_configured() -> bool:
    return bool(os.environ.get("OPJAX_DSPARK_CAPTURE_ROOT"))


def capture_static_tensor(name: str, value: torch.Tensor) -> None:
    root_value = os.environ.get("OPJAX_DSPARK_CAPTURE_ROOT")
    if not root_value:
        return
    _write_tensor(
        root=Path(root_value),
        session="static",
        name=name,
        value=value,
    )


def load_target_feature_override() -> torch.Tensor | None:
    path_value = os.environ.get("OPJAX_DSPARK_TARGET_FEATURE_OVERRIDE")
    if not path_value:
        return None
    path = Path(path_value)
    if not path.is_file():
        raise RuntimeError(f"DSPARK_TARGET_FEATURE_OVERRIDE_MISSING:{path}")
    array = np.load(path, allow_pickle=False)
    if array.ndim == 3 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 2:
        raise RuntimeError(f"DSPARK_TARGET_FEATURE_OVERRIDE_SHAPE:{array.shape}")
    return torch.from_numpy(np.array(array, copy=True))
