from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from opjax.pallas.laguna_serving_native import serving_native_sample_dirs
from opjax.remote.laguna_serving_native_modal import (
    _fixed_request_payload,
    _preserve_or_archive_profile,
    _validate_prefix_cache_start,
)


def test_fixed_request_binds_cache_salt() -> None:
    payload = _fixed_request_payload(
        np.arange(32, dtype=np.int32),
        16,
        cache_salt="a" * 64,
    )
    assert payload["prompt"] == list(range(16))
    assert payload["cache_salt"] == "a" * 64


def test_complete_profile_survives_incomplete_shard_resume(tmp_path: Path) -> None:
    profile_root = tmp_path / "profiles"
    profile_root.mkdir()
    trace = profile_root / "trace.json"
    trace.write_text('{"name":"cudaLaunchKernel"}', encoding="utf-8")
    receipt = _preserve_or_archive_profile(
        run_root=tmp_path, profile_root=profile_root
    )
    assert receipt is not None
    assert receipt["cuda_launch_events"] == 1
    assert trace.is_file()
    assert not (tmp_path / "attempts").exists()


def test_prefix_cache_start_uses_observed_aligned_reuse() -> None:
    _validate_prefix_cache_start(
        prefix_index=0, processed_start=0, previous_end=0
    )
    _validate_prefix_cache_start(
        prefix_index=1, processed_start=208, previous_end=231
    )

    for processed_start in (0, 209, 224, 240):
        with pytest.raises(RuntimeError, match="SERVING_NATIVE_PREFIX_CACHE_START"):
            _validate_prefix_cache_start(
                prefix_index=1,
                processed_start=processed_start,
                previous_end=231,
            )


def test_cache_samples_are_globally_ordered_by_prompt_id(tmp_path: Path) -> None:
    for shard, prompt_id in (("train-000-of-002", "z"), ("train-001-of-002", "a")):
        sample = (
            tmp_path
            / "rebuilt-fixed-shards"
            / shard
            / "policies"
            / "first-causal-observation-wins"
            / "samples"
            / "train"
            / prompt_id
        )
        sample.mkdir(parents=True)

    assert [path.name for path in serving_native_sample_dirs(tmp_path, "train")] == [
        "a",
        "z",
    ]
