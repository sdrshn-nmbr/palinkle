from __future__ import annotations

import json
from pathlib import Path

from opjax.remote.laguna_vllm_entrypoint import (
    _checkpoint_identity,
    _find_vllm_utils_path,
    _patch_dflash_source_alignment,
    _prepare_dspark_snapshot,
    DFLASH_SAMPLE_REPLACEMENT,
    DFLASH_SAMPLE_SOURCE,
)


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
    prepared = _prepare_dspark_snapshot(
        root=tmp_path / "prepared", model=str(source), revision=None
    )
    normalized = json.loads((prepared / "config.json").read_text())
    assert normalized["model_type"] == "laguna"
    assert normalized["swa_rope_parameters"]["rope_theta"] == 500000.0
    assert (prepared / "model.safetensors").resolve() == (
        source / "model.safetensors"
    ).resolve()


def test_checkpoint_identity_binds_config_and_weights(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text("{}")
    (checkpoint / "model.safetensors").write_bytes(b"weights")
    arguments = [
        "target",
        "--speculative-config",
        json.dumps({"model": str(checkpoint), "method": "dflash"}),
    ]
    first = _checkpoint_identity(arguments)
    assert first is not None
    assert first["files"].keys() == {"config.json", "model.safetensors"}
    (checkpoint / "model.safetensors").write_bytes(b"different")
    second = _checkpoint_identity(arguments)
    assert second is not None
    assert first["sha256"] != second["sha256"]


def test_dflash_runtime_alignment_patch_is_exact_and_idempotent(
    tmp_path: Path,
) -> None:
    source = tmp_path / "utils.py"
    source.write_text(f"prefix\n{DFLASH_SAMPLE_SOURCE}suffix\n")
    first = _patch_dflash_source_alignment(source)
    assert first["state"] == "applied"
    assert DFLASH_SAMPLE_SOURCE not in source.read_text()
    assert DFLASH_SAMPLE_REPLACEMENT in source.read_text()
    second = _patch_dflash_source_alignment(source)
    assert second["state"] == "already_applied"
    assert second["after_sha256"] == first["after_sha256"]


def test_dflash_runtime_alignment_patch_fails_on_unknown_source(
    tmp_path: Path,
) -> None:
    source = tmp_path / "utils.py"
    source.write_text("unknown source")
    try:
        _patch_dflash_source_alignment(source)
    except RuntimeError as exc:
        assert "LAGUNA_DFLASH_ALIGNMENT_SOURCE_MISMATCH" in str(exc)
    else:
        raise AssertionError("unknown vLLM source must fail closed")


def test_vllm_source_discovery_uses_launcher_interpreter(tmp_path: Path) -> None:
    prefix = tmp_path / "runtime"
    launcher = prefix / "bin" / "vllm"
    interpreter = prefix / "bin" / "python"
    source = (
        prefix
        / "lib"
        / "python3.12"
        / "site-packages"
        / "vllm"
        / "v1"
        / "spec_decode"
        / "utils.py"
    )
    launcher.parent.mkdir(parents=True)
    source.parent.mkdir(parents=True)
    interpreter.write_text("")
    launcher.write_text(f"#!{interpreter}\n")
    source.write_text(DFLASH_SAMPLE_SOURCE)
    assert _find_vllm_utils_path(launcher) == source


def test_vllm_source_discovery_preserves_symlink_prefix(tmp_path: Path) -> None:
    system = tmp_path / "system"
    system_python = system / "bin" / "python"
    system_python.parent.mkdir(parents=True)
    system_python.write_text("")
    prefix = tmp_path / "runtime"
    interpreter = prefix / "bin" / "python"
    interpreter.parent.mkdir(parents=True)
    interpreter.symlink_to(system_python)
    launcher = prefix / "bin" / "vllm"
    launcher.write_text(f"#!{interpreter}\n")
    source = (
        prefix
        / "lib"
        / "python3.12"
        / "dist-packages"
        / "vllm"
        / "v1"
        / "spec_decode"
        / "utils.py"
    )
    source.parent.mkdir(parents=True)
    source.write_text(DFLASH_SAMPLE_SOURCE)
    assert _find_vllm_utils_path(launcher) == source
