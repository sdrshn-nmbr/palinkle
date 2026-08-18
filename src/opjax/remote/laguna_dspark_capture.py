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
_ACTIVE_ROUNDS: dict[str, int] = {}


def capture_step(value: torch.Tensor, step: int) -> torch.Tensor:
    if step < 0:
        raise RuntimeError(f"LAGUNA_CAPTURE_STEP_RANGE:{step}:{tuple(value.shape)}")
    if value.ndim == 2:
        if step >= value.shape[0]:
            raise RuntimeError(f"LAGUNA_CAPTURE_STEP_RANGE:{step}:{tuple(value.shape)}")
        return value[step : step + 1, :]
    if value.ndim == 3:
        if step >= value.shape[1]:
            raise RuntimeError(f"LAGUNA_CAPTURE_STEP_RANGE:{step}:{tuple(value.shape)}")
        return value[:, step, :]
    raise RuntimeError(f"LAGUNA_CAPTURE_STEP_RANK:{tuple(value.shape)}")


def logical_context_kv(
    kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    *,
    sequence_length: int,
    query_length: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if kv_cache.ndim != 4 or block_table.ndim != 1:
        raise RuntimeError(
            f"LAGUNA_CAPTURE_KV_LAYOUT:{tuple(kv_cache.shape)}:"
            f"{tuple(block_table.shape)}"
        )
    context_length = sequence_length - query_length
    if context_length < 0:
        raise RuntimeError(
            f"LAGUNA_CAPTURE_CONTEXT_LENGTH:{sequence_length}:{query_length}"
        )
    block_size = kv_cache.shape[2]
    block_count = (context_length + block_size - 1) // block_size
    physical_blocks = block_table[:block_count].to(dtype=torch.long)
    if physical_blocks.numel() != block_count or bool((physical_blocks < 0).any()):
        raise RuntimeError("LAGUNA_CAPTURE_BLOCK_TABLE_INVALID")
    packed = (
        kv_cache.index_select(0, physical_blocks)
        .permute(0, 2, 1, 3)
        .reshape(-1, kv_cache.shape[1], kv_cache.shape[3])[:context_length]
    )
    if packed.shape[-1] % 2:
        raise RuntimeError(f"LAGUNA_CAPTURE_KV_WIDTH:{packed.shape[-1]}")
    return packed.chunk(2, dim=-1)


def validate_single_request_attention_layout(
    query_start_loc: torch.Tensor,
    sequence_lengths: torch.Tensor,
    block_table: torch.Tensor,
    slot_mapping: torch.Tensor,
    *,
    query_length: int,
    block_size: int,
) -> int:
    starts = query_start_loc.reshape(-1)
    lengths = sequence_lengths.reshape(-1)
    if starts.numel() != 2 or lengths.numel() != 1 or block_table.ndim != 2:
        raise RuntimeError("LAGUNA_CAPTURE_ATTENTION_METADATA_BATCH_INVALID")
    observed_query_length = int((starts[1] - starts[0]).item())
    if (
        int(starts[0].item()) != 0
        or int(starts[1].item()) != query_length
        or observed_query_length != query_length
        or slot_mapping.numel() != query_length
    ):
        raise RuntimeError(
            "LAGUNA_CAPTURE_QUERY_LAYOUT_INVALID:"
            f"{observed_query_length}:{query_length}:{slot_mapping.numel()}"
        )
    sequence_length = int(lengths[0].item())
    context_length = sequence_length - query_length
    if context_length < 0:
        raise RuntimeError(
            f"LAGUNA_CAPTURE_CONTEXT_LENGTH:{sequence_length}:{query_length}"
        )
    logical_slots = torch.arange(
        context_length,
        sequence_length,
        device=block_table.device,
        dtype=torch.long,
    )
    logical_blocks = torch.div(logical_slots, block_size, rounding_mode="floor")
    offsets = logical_slots.remainder(block_size)
    physical_blocks = block_table[0].index_select(0, logical_blocks)
    expected_slots = physical_blocks * block_size + offsets
    if not torch.equal(slot_mapping.reshape(-1).to(torch.long), expected_slots):
        raise RuntimeError("LAGUNA_CAPTURE_SLOT_MAPPING_INVALID")
    return context_length


def _control() -> tuple[Path, dict[str, Any]] | None:
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
    return root, value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_tensor(
    *, root: Path, session: str, name: str, value: torch.Tensor, round_id: int | None
) -> None:
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
            "round": round_id,
            "sha256": _sha256(path),
        }
        ledger = session_root / "ledger.jsonl"
        with ledger.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def capture_tensor(name: str, value: torch.Tensor) -> None:
    active = _control()
    if active is None:
        return
    root, control = active
    session = control["session"]
    with _LOCK:
        round_id = _ACTIVE_ROUNDS.get(session)
    if round_id is None:
        raise RuntimeError(f"DSPARK_CAPTURE_ROUND_NOT_STARTED:{session}:{name}")
    _write_tensor(
        root=root, session=session, name=name, value=value, round_id=round_id
    )


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
        round_id=None,
    )


def capture_static_metadata(name: str, value: dict[str, Any]) -> None:
    root_value = os.environ.get("OPJAX_DSPARK_CAPTURE_ROOT")
    if not root_value:
        return
    static_root = Path(root_value) / "static"
    static_root.mkdir(parents=True, exist_ok=True)
    path = static_root / f"{name}.json"
    temporary = path.with_suffix(".tmp")
    payload = json.dumps(value, indent=2, sort_keys=True) + "\n"
    temporary.write_text(payload, encoding="utf-8")
    os.replace(temporary, path)


def begin_capture_round() -> int | None:
    active = _control()
    if active is None:
        return None
    session = active[1]["session"]
    with _LOCK:
        round_id = _ACTIVE_ROUNDS.get(session, -1) + 1
        _ACTIVE_ROUNDS[session] = round_id
    return round_id


def load_target_feature_override() -> torch.Tensor | None:
    active = _control()
    path_value = None
    if active is not None:
        control = active[1]
        candidate = control.get("target_feature_override")
        candidates = control.get("target_feature_overrides")
        if candidate is not None and candidates is not None:
            raise RuntimeError("DSPARK_TARGET_FEATURE_OVERRIDE_AMBIGUOUS")
        if candidates is not None:
            if not isinstance(candidates, list) or not all(
                isinstance(item, str) for item in candidates
            ):
                raise RuntimeError("DSPARK_TARGET_FEATURE_OVERRIDES_CONTROL_INVALID")
            with _LOCK:
                index = _ACTIVE_ROUNDS.get(control["session"])
            if index is None:
                raise RuntimeError("DSPARK_TARGET_FEATURE_OVERRIDE_ROUND_MISSING")
            if index >= len(candidates):
                if control.get("allow_native_after_override_exhaustion") is True:
                    return None
                raise RuntimeError(
                    "DSPARK_TARGET_FEATURE_OVERRIDE_EXHAUSTED:"
                    f"{control['session']}:{index}"
                )
            candidate = candidates[index]
        if candidate is not None and not isinstance(candidate, str):
            raise RuntimeError("DSPARK_TARGET_FEATURE_OVERRIDE_CONTROL_INVALID")
        path_value = candidate
    if path_value is None:
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
