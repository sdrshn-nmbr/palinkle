from __future__ import annotations

import json
from pathlib import Path

from opjax.remote.laguna_vllm_entrypoint import _prepare_dspark_snapshot


def test_prepare_dspark_snapshot_accepts_bound_local_checkpoint(tmp_path: Path) -> None:
    source = tmp_path / "checkpoint"
    source.mkdir()
    config = {
        "architectures": ["LagunaDSparkModel"],
        "model_type": "laguna_dspark",
        "vocab_size": 100352,
        "block_size": 16,
        "proposal_length": 15,
        "mask_token_id": 12,
        "num_target_layers": 40,
        "target_layer_ids": [1, 13, 25, 33, 39],
        "draft_causal": True,
        "rope_parameters": {"rope_theta": 500000.0, "rope_type": "default"},
    }
    (source / "config.json").write_text(json.dumps(config))
    (source / "model.safetensors").write_bytes(b"bound-checkpoint")
    prepared = _prepare_dspark_snapshot(root=tmp_path / "prepared", model=str(source), revision=None)
    normalized = json.loads((prepared / "config.json").read_text())
    assert normalized["model_type"] == "laguna"
    assert normalized["swa_rope_parameters"]["rope_theta"] == 500000.0
    assert (prepared / "model.safetensors").resolve() == (
        source / "model.safetensors"
    ).resolve()
