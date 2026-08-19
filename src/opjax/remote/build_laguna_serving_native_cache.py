from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

from deepspec.data import CacheCollator
from deepspec.data.target_cache_dataset import (
    CacheDataset,
    LocalTargetCacheWriter,
    build_target_cache_manifest,
    write_target_cache_manifest,
)
from opjax.pallas.laguna_dspark_conformance import canonical_sha256
from opjax.pallas.laguna_serving_native import (
    serving_native_sample_dirs,
    validate_sample,
)


TARGET_LAYER_IDS = [1, 13, 25, 33, 39]
HIDDEN_SIZE = 2_048
TARGET_MODEL = "poolside/Laguna-XS-2.1"
TARGET_REVISION = "e9df9a59996d790b94b70f3fef343fe1d9e34bdf"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_bfloat16(path: Path) -> torch.Tensor:
    value = np.load(path, allow_pickle=False)
    if value.dtype != np.uint16:
        raise RuntimeError(f"SERVING_NATIVE_CACHE_BFLOAT16_INVALID:{path}:{value.dtype}")
    return torch.from_numpy(value).view(torch.bfloat16)


def build_cache(*, capture_root: Path, split: str, output: Path) -> dict[str, object]:
    release_path = capture_root / "release.json"
    if not release_path.is_file():
        raise RuntimeError(f"SERVING_NATIVE_CACHE_RELEASE_MISSING:{capture_root}")
    release = json.loads(release_path.read_text(encoding="utf-8"))
    if canonical_sha256(
        {key: value for key, value in release.items() if key != "release_sha256"}
    ) != release.get("release_sha256"):
        raise RuntimeError("SERVING_NATIVE_CACHE_RELEASE_HASH_INVALID")
    if output.exists():
        raise RuntimeError(f"SERVING_NATIVE_CACHE_OUTPUT_EXISTS:{output}")
    output.mkdir(parents=True)
    writer = LocalTargetCacheWriter(
        rank_dir=str(output),
        max_shard_bytes=2 * 1024**3,
    )
    sample_dirs = serving_native_sample_dirs(capture_root, split)
    release_records = release["splits"][split]["records"]
    if (
        release["splits"][split].get("feature_selection_policy")
        != "first_causal_observation_wins"
    ):
        raise RuntimeError(f"SERVING_NATIVE_CACHE_POLICY_INVALID:{split}")
    expected_count = release["splits"][split]["record_count"]
    if len(sample_dirs) != expected_count:
        raise RuntimeError(
            f"SERVING_NATIVE_CACHE_RELEASE_COUNT:{split}:{len(sample_dirs)}:{expected_count}"
        )
    expected_ids = [record["id"] for record in release_records]
    observed_ids = [path.name for path in sample_dirs]
    if observed_ids != sorted(expected_ids):
        raise RuntimeError(f"SERVING_NATIVE_CACHE_RELEASE_IDS:{split}")
    release_by_id = {record["id"]: record for record in release_records}
    source = []
    try:
        for sample_id, sample_dir in enumerate(sample_dirs):
            validate_sample(sample_dir)
            manifest_path = sample_dir / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            release_record = release_by_id[sample_dir.name]
            if (
                manifest.get("manifest_sha256")
                != release_record.get("manifest_sha256")
                or _sha256(manifest_path)
                != release_record.get("manifest_file_sha256")
                or manifest.get("metadata", {}).get("feature_selection_policy")
                != "first_causal_observation_wins"
                or manifest.get("metadata", {}).get("rebuild_driver_sha256")
                != release["splits"][split].get("rebuild_driver_sha256")
                or manifest.get("metadata", {}).get(
                    "reconstruction_source_sha256"
                )
                != release["splits"][split].get(
                    "reconstruction_source_sha256"
                )
            ):
                raise RuntimeError(
                    f"SERVING_NATIVE_CACHE_RELEASE_SAMPLE:{split}:{sample_dir.name}"
                )
            files = manifest["files"]
            input_ids = torch.from_numpy(
                np.load(sample_dir / files["input_ids"]["path"], allow_pickle=False)
            )
            attention_mask = torch.from_numpy(
                np.load(
                    sample_dir / files["attention_mask"]["path"], allow_pickle=False
                )
            )
            loss_mask = torch.from_numpy(
                np.load(sample_dir / files["loss_mask"]["path"], allow_pickle=False)
            )
            hidden = _load_bfloat16(
                sample_dir / files["target_hidden_states"]["path"]
            )
            last_hidden = _load_bfloat16(
                sample_dir / files["target_last_hidden_states"]["path"]
            )
            writer.write_sample(
                sample_id=sample_id,
                input_ids=input_ids,
                attention_mask=attention_mask,
                loss_mask=loss_mask,
                target_hidden_states=hidden,
                target_last_hidden_states=last_hidden,
            )
            source.append(
                {
                    "prompt_id": sample_dir.name,
                    "manifest_sha256": manifest["manifest_sha256"],
                    "manifest_file_sha256": _sha256(manifest_path),
                }
            )
    finally:
        writer.close()
    local_index = output / "samples.local.idx"
    os.replace(local_index, output / "samples.idx")
    shards = []
    for shard_id, local_path in enumerate(sorted(output.glob("shard-local-*.bin"))):
        name = f"shard-{shard_id:05d}.bin"
        os.replace(local_path, output / name)
        shards.append({"shard_id": shard_id, "file_name": name})
    manifest = build_target_cache_manifest(
        num_samples=len(source),
        shards=shards,
        target_layer_ids=TARGET_LAYER_IDS,
        hidden_size=HIDDEN_SIZE,
        extra_fields={
            "kind": "opjax_laguna_serving_native_target_cache",
            "split": split,
            "target_model_name_or_path": TARGET_MODEL,
            "target_revision": TARGET_REVISION,
            "capture_root": str(capture_root),
            "capture_release_sha256": release["release_sha256"],
            "capture_release_file_sha256": _sha256(release_path),
            "source_samples": source,
        },
    )
    write_target_cache_manifest(output_dir=str(output), manifest=manifest)
    dataset = CacheDataset(cache_dir=str(output))
    try:
        if len(dataset) != len(source):
            raise RuntimeError("SERVING_NATIVE_CACHE_ROUNDTRIP_COUNT_INVALID")
        collator = CacheCollator()
        for index, sample_dir in enumerate(sample_dirs):
            expected = np.load(sample_dir / "input_ids.npy", allow_pickle=False)
            observed = dataset[index]
            if not np.array_equal(observed["input_ids"].numpy(), expected):
                raise RuntimeError(
                    f"SERVING_NATIVE_CACHE_ROUNDTRIP_TOKENS_INVALID:{index}"
                )
            expected_loss = np.load(sample_dir / "loss_mask.npy", allow_pickle=False)
            if not np.array_equal(observed["loss_mask"].numpy(), expected_loss):
                raise RuntimeError(
                    f"SERVING_NATIVE_CACHE_ROUNDTRIP_MASK_INVALID:{index}"
                )
            expected_hidden = np.load(
                sample_dir / "target_hidden_states.npy", allow_pickle=False
            )
            expected_last = np.load(
                sample_dir / "target_last_hidden_states.npy", allow_pickle=False
            )
            if not np.array_equal(
                observed["target_hidden_states"].view(torch.uint16).numpy(),
                expected_hidden,
            ):
                raise RuntimeError(
                    f"SERVING_NATIVE_CACHE_ROUNDTRIP_HIDDEN_INVALID:{index}"
                )
            if not np.array_equal(
                observed["target_last_hidden_states"].view(torch.uint16).numpy(),
                expected_last,
            ):
                raise RuntimeError(
                    f"SERVING_NATIVE_CACHE_ROUNDTRIP_LAST_HIDDEN_INVALID:{index}"
                )
            batch = collator([observed])
            if not torch.equal(
                batch["attention_mask"], torch.ones_like(batch["attention_mask"])
            ):
                raise RuntimeError(
                    f"SERVING_NATIVE_CACHE_ROUNDTRIP_ATTENTION_INVALID:{index}"
                )
    finally:
        dataset.close()
    files = {
        str(path.relative_to(output)): _sha256(path)
        for path in sorted(output.iterdir())
        if path.is_file()
    }
    receipt: dict[str, object] = {
        "schema_version": 1,
        "split": split,
        "samples": len(source),
        "capture_release_sha256": release["release_sha256"],
        "feature_selection_policy": "first_causal_observation_wins",
        "target_model_name_or_path": TARGET_MODEL,
        "target_revision": TARGET_REVISION,
        "files": files,
    }
    receipt["receipt_sha256"] = hashlib.sha256(
        json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    (output / "opjax-receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture-root", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "calibration", "heldout"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build_cache(**vars(args)), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
