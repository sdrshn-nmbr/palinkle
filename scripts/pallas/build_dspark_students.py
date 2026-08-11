from __future__ import annotations

import json
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


SOURCE = Path("/tmp/opjax-dspark-control-736501c3")
OUTPUT = Path("/tmp/opjax-dspark-students-20260810")
SOURCE_TARGET_LAYERS = [1, 6, 12, 17, 23, 28, 34, 39]
TARGET_LAYERS = [6, 23, 39]
HIDDEN_SIZE = 4096


def build(name: str, source_layers: list[int], markov_rank: int) -> None:
    destination = OUTPUT / name
    destination.mkdir(parents=True, exist_ok=False)
    config = json.loads((SOURCE / "config.json").read_text())
    config["num_hidden_layers"] = len(source_layers)
    config["max_window_layers"] = len(source_layers)
    config["layer_types"] = ["full_attention"] * len(source_layers)
    config["dflash_config"]["target_layer_ids"] = TARGET_LAYERS
    config["dflash_config"]["markov_rank"] = markov_rank
    config["markov_rank"] = markov_rank
    (destination / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    for filename in ["dflash.py", "dspark.py"]:
        shutil.copy2(SOURCE / filename, destination / filename)

    tensors: dict[str, torch.Tensor] = {}
    source_to_destination = {
        source_layer: destination_layer
        for destination_layer, source_layer in enumerate(source_layers)
    }
    with safe_open(SOURCE / "model.safetensors", framework="pt", device="cpu") as handle:
        metadata = handle.metadata()
        for key in handle.keys():
            tensor = handle.get_tensor(key)
            if key.startswith("layers."):
                source_layer = int(key.split(".", 2)[1])
                if source_layer not in source_to_destination:
                    continue
                destination_layer = source_to_destination[source_layer]
                destination_key = key.replace(
                    f"layers.{source_layer}.",
                    f"layers.{destination_layer}.",
                    1,
                )
                tensors[destination_key] = tensor.contiguous()
                continue
            if key == "fc.weight":
                groups = tensor.reshape(HIDDEN_SIZE, len(SOURCE_TARGET_LAYERS), HIDDEN_SIZE)
                indices = [SOURCE_TARGET_LAYERS.index(layer) for layer in TARGET_LAYERS]
                tensors[key] = groups[:, indices, :].reshape(HIDDEN_SIZE, -1).contiguous()
                continue
            if key == "markov_head.markov_w1.weight":
                tensors[key] = tensor[:, :markov_rank].contiguous()
                continue
            if key == "markov_head.markov_w2.weight":
                tensors[key] = tensor[:, :markov_rank].contiguous()
                continue
            if key == "confidence_head.proj.weight":
                hidden = tensor[:, :HIDDEN_SIZE]
                markov = tensor[:, HIDDEN_SIZE : HIDDEN_SIZE + markov_rank]
                tensors[key] = torch.cat([hidden, markov], dim=1).contiguous()
                continue
            tensors[key] = tensor.contiguous()
    save_file(tensors, destination / "model.safetensors", metadata=metadata)
    parameter_count = sum(tensor.numel() for tensor in tensors.values())
    manifest = {
        "name": name,
        "source_revision": "736501c3901cfc6bbb53ba382781eb0e5d9ad66a",
        "source_layers": source_layers,
        "target_layer_ids": TARGET_LAYERS,
        "markov_rank": markov_rank,
        "parameter_count": parameter_count,
    }
    (destination / "compression-manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(json.dumps(manifest))


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=False)
    build("dspark-500m", source_layers=[0, 3], markov_rank=128)
    build("dspark-250m", source_layers=[0], markov_rank=64)


if __name__ == "__main__":
    main()

