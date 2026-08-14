from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
from safetensors.torch import load_file
from transformers import AutoConfig, PretrainedConfig

from deepspec.modeling.dspark.laguna import LagunaDSparkModel, build_draft_config


TARGET = "poolside/Laguna-XS-2.1"
TARGET_REVISION = "e9df9a59996d790b94b70f3fef343fe1d9e34bdf"
DFLASH = "poolside/Laguna-XS-2.1-DFlash"
DFLASH_REVISION = "5c36361aab23c8ed3afbd079c10c426b677bc607"
TARGET_KEYS = {"embed_tokens.weight", "lm_head.weight"}
DSPARK_KEYS = {
    "markov_head.markov_w1.weight",
    "markov_head.markov_w2.weight",
    "confidence_head.proj.weight",
    "confidence_head.proj.bias",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _target_tensors(target_dir: Path) -> dict[str, torch.Tensor]:
    names = ("model.embed_tokens.weight", "lm_head.weight")
    index = json.loads(
        (target_dir / "model.safetensors.index.json").read_text(encoding="utf-8")
    )["weight_map"]
    result = {}
    for name in names:
        with safe_open(
            target_dir / index[name], framework="pt", device="cpu"
        ) as handle:
            result[name] = handle.get_tensor(name)
    return result


def initialize(*, arm: str, output_root: Path) -> dict[str, object]:
    if arm not in {"dflash", "dspark"}:
        raise ValueError(f"LAGUNA_SPECULATOR_ARM_INVALID:{arm}")
    source_dir = Path(
        snapshot_download(
            DFLASH,
            revision=DFLASH_REVISION,
            allow_patterns=["config.json", "*.safetensors"],
        )
    )
    target_dir = Path(
        snapshot_download(
            TARGET,
            revision=TARGET_REVISION,
            allow_patterns=["config.json", "*.safetensors", "*.json"],
        )
    )
    source_payload, _ = PretrainedConfig.get_config_dict(source_dir)
    target_config = AutoConfig.from_pretrained(target_dir, trust_remote_code=True)
    dspark = arm == "dspark"
    model_args = {
        "target_layer_ids": source_payload["dflash_config"]["target_layer_ids"],
        "block_size": source_payload["dflash_config"]["block_size"],
        "proposal_length": source_payload["dflash_config"]["block_size"] - 1,
        "mask_token_id": source_payload["dflash_config"]["mask_token_id"],
        "num_anchors": 64,
        "markov_rank": 256 if dspark else 0,
        "markov_head_type": "vanilla",
        "confidence_head_alpha": 1.0 if dspark else 0.0,
        "confidence_head_with_markov": dspark,
    }
    config = build_draft_config(
        target_config=target_config,
        model_args=model_args,
        source_config=source_payload,
    )
    model = LagunaDSparkModel(config).to(dtype=torch.bfloat16)
    source_state = load_file(source_dir / "model.safetensors", device="cpu")
    expected_missing = DSPARK_KEYS if dspark else set()
    expected_source = set(model.state_dict()) - TARGET_KEYS - expected_missing
    if set(source_state) != expected_source:
        raise RuntimeError(
            "LAGUNA_DFLASH_BACKBONE_MISMATCH:"
            f"missing={sorted(expected_source - set(source_state))}:"
            f"unexpected={sorted(set(source_state) - expected_source)}"
        )
    target = _target_tensors(target_dir)
    combined = dict(source_state)
    combined["embed_tokens.weight"] = target["model.embed_tokens.weight"]
    combined["lm_head.weight"] = target["lm_head.weight"]
    incompatible = model.load_state_dict(combined, strict=False)
    if set(incompatible.missing_keys) != expected_missing:
        raise RuntimeError(
            f"LAGUNA_INITIALIZER_MISSING_INVALID:{incompatible.missing_keys}"
        )
    if incompatible.unexpected_keys:
        raise RuntimeError(
            f"LAGUNA_INITIALIZER_UNEXPECTED:{incompatible.unexpected_keys}"
        )
    model.set_embedding_head_trainable(False)
    output = output_root / arm
    output.mkdir(parents=True, exist_ok=False)
    model.save_pretrained(output, safe_serialization=True)
    checkpoint = output / "model.safetensors"
    manifest: dict[str, object] = {
        "schema_version": 1,
        "arm": arm,
        "target": {"repo": TARGET, "revision": TARGET_REVISION},
        "source": {"repo": DFLASH, "revision": DFLASH_REVISION},
        "source_tensor_count": len(source_state),
        "randomly_initialized": sorted(expected_missing),
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "trainable_parameters": sum(
            p.numel() for p in model.parameters() if p.requires_grad
        ),
        "checkpoint_sha256": _sha256(checkpoint),
    }
    (output / "initialization.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    reloaded = LagunaDSparkModel.from_pretrained(output, dtype=torch.bfloat16)
    if set(reloaded.state_dict()) != set(model.state_dict()):
        raise RuntimeError("LAGUNA_INITIALIZER_RELOAD_MISMATCH")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=("dflash", "dspark"), required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(initialize(arm=args.arm, output_root=args.output_root), indent=2))


if __name__ == "__main__":
    main()
