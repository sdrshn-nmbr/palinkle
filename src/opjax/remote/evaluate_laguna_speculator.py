from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile, record_function
from torch.utils.data import DataLoader

from deepspec.data import CacheCollator, CacheDataset
from deepspec.modeling.dspark.laguna import LagunaDSparkModel
from deepspec.modeling.dspark.loss import (
    _collect_local_terms,
    _compute_accept_rate_3d,
)


def _forward(model, batch):
    return model(
        input_ids=batch["input_ids"],
        target_hidden_states=batch["target_hidden_states"],
        loss_mask=batch["loss_mask"],
        target_last_hidden_states=batch["target_last_hidden_states"],
    )


def evaluate(
    *, checkpoint: Path, cache: Path, output: Path, num_anchors: int, seed: int
) -> dict[str, object]:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda")
    model = LagunaDSparkModel.from_pretrained(
        checkpoint,
        dtype=torch.bfloat16,
        attn_implementation="flex_attention",
    ).to(device).eval()
    model.num_anchors = num_anchors
    dataset = CacheDataset(cache_dir=str(cache))
    loader = DataLoader(dataset, batch_size=1, collate_fn=CacheCollator(), shuffle=False)
    totals = {
        "ce_loss_num": 0.0,
        "ce_loss_den": 0.0,
        "l1_loss_num": 0.0,
        "l1_loss_den": 0.0,
        "confidence_loss_num": 0.0,
        "confidence_loss_den": 0.0,
    }
    accept_sums = None
    accept_counts = None
    greedy_prefix_sum = 0.0
    greedy_blocks = 0.0
    tau_prob_sum = 0.0
    calibration = [
        {"count": 0.0, "predicted_sum": 0.0, "observed_sum": 0.0}
        for _ in range(10)
    ]
    batches = 0
    first_batch = None
    with torch.inference_mode():
        for cpu_batch in loader:
            batch = {key: value.to(device) for key, value in cpu_batch.items()}
            batch["input_ids"] = batch["input_ids"].long()
            if first_batch is None:
                first_batch = batch
            outputs = _forward(model, batch)
            terms, _ = _collect_local_terms(
                outputs=outputs,
                loss_decay_gamma=4.0,
                l1_loss_alpha=0.9,
            )
            for key in totals:
                totals[key] += float(terms[key].item())
            rates = _compute_accept_rate_3d(
                outputs=outputs,
                aligned_target_logits=outputs.aligned_target_logits,
            )
            if rates is None:
                raise RuntimeError("LAGUNA_HELDOUT_TARGET_LOGITS_MISSING")
            valid = outputs.eval_mask.to(torch.float32)
            position_sums = (rates * valid).sum(dim=(0, 1))
            position_counts = valid.sum(dim=(0, 1))
            accept_sums = position_sums if accept_sums is None else accept_sums + position_sums
            accept_counts = (
                position_counts
                if accept_counts is None
                else accept_counts + position_counts
            )
            valid_blocks = outputs.block_keep_mask & outputs.eval_mask.any(dim=-1)
            tau_prob_sum += float(
                ((rates * valid).cumprod(dim=-1).sum(dim=-1) * valid_blocks).sum().item()
            )
            greedy = outputs.draft_logits.argmax(dim=-1).eq(outputs.target_ids) & outputs.eval_mask
            greedy_prefix_sum += float(
                (greedy.to(torch.float32).cumprod(dim=-1).sum(dim=-1) * valid_blocks).sum().item()
            )
            greedy_blocks += float(valid_blocks.sum().item())
            if outputs.confidence_pred is not None:
                predicted = outputs.confidence_pred.sigmoid()
                for index in range(10):
                    low = index / 10
                    high = (index + 1) / 10
                    selected = valid.bool() & (predicted >= low) & (
                        predicted <= high if index == 9 else predicted < high
                    )
                    count = float(selected.sum().item())
                    calibration[index]["count"] += count
                    calibration[index]["predicted_sum"] += float(predicted[selected].sum().item())
                    calibration[index]["observed_sum"] += float(rates[selected].sum().item())
            batches += 1
    if first_batch is None or accept_sums is None or accept_counts is None:
        raise RuntimeError("LAGUNA_HELDOUT_CACHE_EMPTY")
    torch.cuda.synchronize()
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
    ) as profiler:
        with torch.inference_mode(), record_function("laguna_speculator_heldout_forward"):
            _forward(model, first_batch)
        torch.cuda.synchronize()
    output.mkdir(parents=True, exist_ok=True)
    profiler.export_chrome_trace(str(output / "torch-trace.json"))
    positions = [
        float(total / max(count, 1.0))
        for total, count in zip(accept_sums.tolist(), accept_counts.tolist())
    ]
    result: dict[str, object] = {
        "schema_version": 1,
        "checkpoint": str(checkpoint),
        "cache": str(cache),
        "batches": batches,
        "seed": seed,
        "loss": {
            "cross_entropy": totals["ce_loss_num"] / max(totals["ce_loss_den"], 1e-9),
            "l1": totals["l1_loss_num"] / max(totals["l1_loss_den"], 1e-9),
            "confidence_bce": totals["confidence_loss_num"]
            / max(totals["confidence_loss_den"], 1e-9),
        },
        "probabilistic_acceptance_by_position": positions,
        "probabilistic_tau": 1.0 + tau_prob_sum / max(greedy_blocks, 1.0),
        "greedy_expected_accepted_tokens": greedy_prefix_sum / max(greedy_blocks, 1.0),
        "greedy_tau": 1.0 + greedy_prefix_sum / max(greedy_blocks, 1.0),
        "valid_blocks": greedy_blocks,
        "confidence_calibration": calibration,
        "trace": "torch-trace.json",
    }
    (output / "evaluation.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-anchors", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(json.dumps(evaluate(**vars(args)), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
