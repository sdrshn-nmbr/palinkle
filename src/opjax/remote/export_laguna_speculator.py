from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

from huggingface_hub import snapshot_download
from safetensors.torch import load_file, save_file

from opjax.remote.initialize_laguna_speculators import DFLASH, DFLASH_REVISION


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def export_dflash(*, checkpoint: Path, output: Path) -> dict[str, object]:
    source = load_file(checkpoint / "model.safetensors", device="cpu")
    target_keys = {"embed_tokens.weight", "lm_head.weight"}
    missing = target_keys - set(source)
    if missing:
        raise RuntimeError(f"LAGUNA_DFLASH_EXPORT_TARGET_KEYS_MISSING:{sorted(missing)}")
    backbone = {key: value for key, value in source.items() if key not in target_keys}
    if len(backbone) != 58:
        raise RuntimeError(f"LAGUNA_DFLASH_EXPORT_TENSOR_COUNT:{len(backbone)}")
    official = Path(
        snapshot_download(
            DFLASH,
            revision=DFLASH_REVISION,
            allow_patterns=["config.json"],
        )
    )
    output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(official / "config.json", output / "config.json")
    save_file(backbone, output / "model.safetensors")
    manifest: dict[str, object] = {
        "schema_version": 1,
        "kind": "opjax_laguna_dflash_export",
        "source_checkpoint": str(checkpoint),
        "source_checkpoint_sha256": _sha256(checkpoint / "model.safetensors"),
        "export_tensor_count": len(backbone),
        "excluded_shared_target_tensors": sorted(target_keys),
        "model_sha256": _sha256(output / "model.safetensors"),
        "config_sha256": _sha256(output / "config.json"),
        "runtime_config_source": {"repo": DFLASH, "revision": DFLASH_REVISION},
    }
    (output / "export.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(export_dflash(**vars(args)), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
