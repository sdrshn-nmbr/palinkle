from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from deepspec.data import CacheCollator, CacheDataset
from deepspec.modeling.dspark.laguna import LagunaDSparkModel
from deepspec.modeling.dspark.loss import _compute_accept_rate_3d


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ece(logits: torch.Tensor, targets: torch.Tensor) -> float:
    probabilities = logits.sigmoid()
    result = torch.tensor(0.0, dtype=torch.float64)
    for index in range(10):
        low = index / 10
        high = (index + 1) / 10
        selected = (probabilities >= low) & (
            probabilities <= high if index == 9 else probabilities < high
        )
        if selected.any():
            result += selected.double().mean() * torch.abs(
                probabilities[selected].mean() - targets[selected].mean()
            )
    return float(result.item())


@torch.inference_mode()
def _collect(
    model: LagunaDSparkModel, cache: Path
) -> tuple[torch.Tensor, torch.Tensor, int]:
    loader = DataLoader(
        CacheDataset(cache_dir=str(cache)),
        batch_size=1,
        collate_fn=CacheCollator(),
        shuffle=False,
    )
    logits: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    batches = 0
    for cpu_batch in loader:
        batch = {key: value.to("cuda") for key, value in cpu_batch.items()}
        batch["input_ids"] = batch["input_ids"].long()
        outputs = model(
            input_ids=batch["input_ids"],
            target_hidden_states=batch["target_hidden_states"],
            loss_mask=batch["loss_mask"],
            target_last_hidden_states=batch["target_last_hidden_states"],
        )
        if outputs.confidence_pred is None:
            raise RuntimeError("LAGUNA_CALIBRATION_CONFIDENCE_MISSING")
        rates = _compute_accept_rate_3d(
            outputs=outputs,
            aligned_target_logits=outputs.aligned_target_logits,
        )
        if rates is None:
            raise RuntimeError("LAGUNA_CALIBRATION_TARGET_LOGITS_MISSING")
        selected = outputs.eval_mask.bool()
        logits.append(outputs.confidence_pred[selected].float().cpu())
        targets.append(rates[selected].float().cpu())
        batches += 1
    if not logits:
        raise RuntimeError("LAGUNA_CALIBRATION_CACHE_EMPTY")
    return torch.cat(logits), torch.cat(targets), batches


def calibrate(*, checkpoint: Path, cache: Path, output: Path) -> dict[str, object]:
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    model = (
        LagunaDSparkModel.from_pretrained(
            checkpoint,
            dtype=torch.bfloat16,
            attn_implementation="flex_attention",
        )
        .to("cuda")
        .eval()
    )
    raw_logits, targets, batches = _collect(model, cache)
    scale_parameter = torch.tensor(0.5413248546, requires_grad=True)
    offset = torch.tensor(0.0, requires_grad=True)
    optimizer = torch.optim.LBFGS(
        [scale_parameter, offset], max_iter=100, line_search_fn="strong_wolfe"
    )

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        scale = F.softplus(scale_parameter)
        loss = F.binary_cross_entropy_with_logits(raw_logits * scale + offset, targets)
        loss.backward()
        return loss

    optimizer.step(closure)
    scale = float(F.softplus(scale_parameter).detach().item())
    resolved_offset = float(offset.detach().item())
    calibrated_logits = raw_logits * scale + resolved_offset
    before_bce = float(F.binary_cross_entropy_with_logits(raw_logits, targets).item())
    after_bce = float(
        F.binary_cross_entropy_with_logits(calibrated_logits, targets).item()
    )
    if after_bce > before_bce + 1e-7:
        raise RuntimeError(f"LAGUNA_CALIBRATION_REGRESSION:{before_bce}:{after_bce}")
    with torch.no_grad():
        projection = model.confidence_head.proj
        projection.weight.mul_(scale)
        projection.bias.mul_(scale).add_(resolved_offset)
    output.mkdir(parents=True, exist_ok=False)
    model.save_pretrained(output, safe_serialization=True)
    result: dict[str, object] = {
        "schema_version": 1,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": _sha256(checkpoint / "model.safetensors"),
        "cache": str(cache.resolve()),
        "batches": batches,
        "points": int(raw_logits.numel()),
        "scale": scale,
        "offset": resolved_offset,
        "before": {"bce": before_bce, "ece": _ece(raw_logits, targets)},
        "after": {"bce": after_bce, "ece": _ece(calibrated_logits, targets)},
        "calibrated_checkpoint_sha256": _sha256(output / "model.safetensors"),
        "seed": 42,
    }
    result["sha256"] = hashlib.sha256(
        json.dumps(result, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    (output / "calibration.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(calibrate(**vars(args)), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
