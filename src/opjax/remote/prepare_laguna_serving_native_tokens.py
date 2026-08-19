from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

from deepspec.data.parser import preprocess_record
from opjax.pallas.laguna_dspark_conformance import canonical_sha256


TARGET_ID = "poolside/Laguna-XS-2.1"
TARGET_REVISION = "e9df9a59996d790b94b70f3fef343fe1d9e34bdf"
MAX_LENGTH = 18_432
EXPECTED_SPLIT_COUNTS = {"train": 102, "calibration": 18, "heldout": 18}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_array(path: Path, value: np.ndarray) -> dict[str, object]:
    np.save(path, value, allow_pickle=False)
    return {
        "path": path.name,
        "sha256": _sha256(path),
        "shape": list(value.shape),
        "dtype": str(value.dtype),
    }


def validate_release(*, output: Path, source_root: Path) -> dict[str, object]:
    release_path = output / "release.json"
    release = json.loads(release_path.read_text(encoding="utf-8"))
    if release.get("release_sha256") != canonical_sha256(
        {key: value for key, value in release.items() if key != "release_sha256"}
    ):
        raise RuntimeError("SERVING_NATIVE_TOKEN_RELEASE_HASH_INVALID")
    source_manifest_path = source_root / "manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    if (
        release.get("source_manifest_sha256")
        != source_manifest.get("manifest_sha256")
        or release.get("source_manifest_file_sha256")
        != _sha256(source_manifest_path)
        or release.get("target")
        != {"repo": TARGET_ID, "revision": TARGET_REVISION}
        or release.get("tokenizer_revision") != TARGET_REVISION
    ):
        raise RuntimeError("SERVING_NATIVE_TOKEN_RELEASE_PROVENANCE_INVALID")
    all_ids: set[str] = set()
    all_trajectories: set[str] = set()
    for split, expected_count in EXPECTED_SPLIT_COUNTS.items():
        split_record = release.get("splits", {}).get(split, {})
        source_path = source_root / f"{split}.jsonl"
        records = split_record.get("records", [])
        if (
            split_record.get("count") != expected_count
            or len(records) != expected_count
            or split_record.get("source_sha256") != _sha256(source_path)
        ):
            raise RuntimeError(f"SERVING_NATIVE_TOKEN_RELEASE_SPLIT_INVALID:{split}")
        for record in records:
            sample_id = record.get("id")
            trajectory = record.get("trajectory")
            sample_root = output / split / str(sample_id)
            manifest_path = sample_root / "manifest.json"
            if (
                not isinstance(sample_id, str)
                or not isinstance(trajectory, str)
                or sample_id in all_ids
                or trajectory in all_trajectories
                or not manifest_path.is_file()
                or _sha256(manifest_path) != record.get("manifest_file_sha256")
            ):
                raise RuntimeError(
                    f"SERVING_NATIVE_TOKEN_RELEASE_RECORD_INVALID:{split}:{sample_id}"
                )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if (
                manifest.get("manifest_sha256") != record.get("manifest_sha256")
                or manifest.get("manifest_sha256")
                != canonical_sha256(
                    {
                        key: value
                        for key, value in manifest.items()
                        if key != "manifest_sha256"
                    }
                )
            ):
                raise RuntimeError(
                    f"SERVING_NATIVE_TOKEN_RELEASE_MANIFEST_INVALID:{split}:{sample_id}"
                )
            for file_record in manifest.get("files", {}).values():
                path = sample_root / file_record["path"]
                if not path.is_file() or _sha256(path) != file_record.get("sha256"):
                    raise RuntimeError(
                        f"SERVING_NATIVE_TOKEN_RELEASE_FILE_INVALID:{split}:{sample_id}"
                    )
            all_ids.add(sample_id)
            all_trajectories.add(trajectory)
    return release


def prepare(*, source_root: Path, output: Path) -> dict[str, object]:
    if output.exists():
        raise RuntimeError(f"SERVING_NATIVE_TOKENS_OUTPUT_EXISTS:{output}")
    output.mkdir(parents=True)
    tokenizer_root = Path(
        snapshot_download(
            TARGET_ID,
            revision=TARGET_REVISION,
            allow_patterns=["*.json", "*.jinja", "*.model", "tokenizer*"],
        )
    )
    if tokenizer_root.name != TARGET_REVISION:
        raise RuntimeError(
            f"SERVING_NATIVE_TOKENIZER_SNAPSHOT:{tokenizer_root.name}:{TARGET_REVISION}"
        )
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_root,
        trust_remote_code=True,
        fix_mistral_regex=True,
    )
    source_manifest_path = source_root / "manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    if source_manifest.get("manifest_sha256") != canonical_sha256(
        {
            key: value
            for key, value in source_manifest.items()
            if key != "manifest_sha256"
        }
    ):
        raise RuntimeError("SERVING_NATIVE_SOURCE_MANIFEST_HASH_INVALID")
    if source_manifest.get("rows") != EXPECTED_SPLIT_COUNTS:
        raise RuntimeError("SERVING_NATIVE_SOURCE_COUNTS_INVALID")
    observed_revision = tokenizer_root.name
    seen_ids: set[str] = set()
    seen_trajectories: set[str] = set()
    seen_token_hashes: set[str] = set()
    splits = {}
    for split in ("train", "calibration", "heldout"):
        source_path = source_root / f"{split}.jsonl"
        source_file = source_manifest["files"][split]
        if (
            source_file.get("path") != source_path.name
            or source_file.get("sha256") != _sha256(source_path)
            or source_file.get("bytes") != source_path.stat().st_size
        ):
            raise RuntimeError(f"SERVING_NATIVE_SOURCE_FILE_INVALID:{split}")
        split_root = output / split
        split_root.mkdir()
        records = []
        for line_number, line in enumerate(
            source_path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            record = json.loads(line)
            sample_id = record.get("id")
            trajectory = record.get("trajectory")
            if (
                not isinstance(sample_id, str)
                or not isinstance(trajectory, str)
                or sample_id in seen_ids
                or trajectory in seen_trajectories
            ):
                raise RuntimeError(
                    f"SERVING_NATIVE_TOKEN_ID_INVALID:{split}:{line_number}"
                )
            parsed = preprocess_record(
                record,
                tokenizer=tokenizer,
                chat_template="laguna_thinking",
                max_length=1_000_000,
            )
            input_ids = parsed["input_ids"].numpy().astype(np.int32)
            attention_mask = parsed["attention_mask"].numpy().astype(np.uint8)
            loss_mask = parsed["loss_mask"].numpy().astype(np.uint8)
            if len(input_ids) > MAX_LENGTH or int(loss_mask.sum()) < 14:
                raise RuntimeError(
                    f"SERVING_NATIVE_TOKEN_LENGTH_INVALID:{sample_id}:"
                    f"{len(input_ids)}:{int(loss_mask.sum())}"
                )
            token_hash = hashlib.sha256(input_ids.tobytes()).hexdigest()
            if token_hash in seen_token_hashes:
                raise RuntimeError(f"SERVING_NATIVE_TOKEN_DUPLICATE:{sample_id}")
            sample_root = split_root / sample_id
            sample_root.mkdir()
            files = {
                "input_ids": _write_array(sample_root / "input_ids.npy", input_ids),
                "attention_mask": _write_array(
                    sample_root / "attention_mask.npy", attention_mask
                ),
                "loss_mask": _write_array(sample_root / "loss_mask.npy", loss_mask),
            }
            manifest: dict[str, object] = {
                "schema_version": 1,
                "kind": "opjax_laguna_serving_native_tokens",
                "split": split,
                "id": sample_id,
                "trajectory": trajectory,
                "task": record["task"],
                "seed": record["seed"],
                "assistant_calls": record["assistant_calls"],
                "tokens": len(input_ids),
                "loss_tokens": int(loss_mask.sum()),
                "token_sha256": token_hash,
                "source_line_sha256": hashlib.sha256(line.encode()).hexdigest(),
                "files": files,
            }
            manifest["manifest_sha256"] = canonical_sha256(manifest)
            manifest_path = sample_root / "manifest.json"
            manifest_path.write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            records.append(
                {
                    "id": sample_id,
                    "trajectory": trajectory,
                    "manifest_sha256": manifest["manifest_sha256"],
                    "manifest_file_sha256": _sha256(manifest_path),
                }
            )
            seen_ids.add(sample_id)
            seen_trajectories.add(trajectory)
            seen_token_hashes.add(token_hash)
        if len(records) != EXPECTED_SPLIT_COUNTS[split]:
            raise RuntimeError(
                f"SERVING_NATIVE_TOKEN_COUNT_INVALID:{split}:{len(records)}"
            )
        splits[split] = {
            "source_sha256": _sha256(source_path),
            "records": records,
            "count": len(records),
        }
    release: dict[str, object] = {
        "schema_version": 1,
        "kind": "opjax_laguna_serving_native_token_release",
        "target": {"repo": TARGET_ID, "revision": TARGET_REVISION},
        "tokenizer_revision": observed_revision,
        "max_length": MAX_LENGTH,
        "source_manifest_sha256": source_manifest["manifest_sha256"],
        "source_manifest_file_sha256": _sha256(source_manifest_path),
        "splits": splits,
    }
    release["release_sha256"] = canonical_sha256(release)
    (output / "release.json").write_text(
        json.dumps(release, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return validate_release(output=output, source_root=source_root)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(**vars(args)), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
