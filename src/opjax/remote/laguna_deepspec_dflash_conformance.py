from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile, record_function
from transformers import AutoTokenizer, DynamicCache

from deepspec.data.parser import encode_chat_messages
from deepspec.eval.dspark.draft_ops import forward_dspark_draft_block
from deepspec.modeling.dspark.laguna import (
    LagunaDSparkModel,
    load_laguna_target_model_strict,
)

from opjax.pallas.laguna_dspark_conformance import canonical_sha256
from opjax.pallas.laguna_speculative import TARGET_REVISION
from opjax.remote.laguna_deepspec_conformance import (
    _capture_target_layers,
    _combined_target_feature,
    _save_tensor,
    _sha256,
)


@torch.inference_mode()
def run_capture(
    *, output_root: Path, prompt: str, target_path: Path, draft_path: Path
) -> dict[str, object]:
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    output_root.mkdir(parents=True, exist_ok=False)
    target_model = (
        load_laguna_target_model_strict(
            target_path,
            trust_remote_code=True,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
        .to("cuda")
        .eval()
    )
    draft_model = (
        LagunaDSparkModel.from_pretrained(
            draft_path,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
        .to("cuda")
        .eval()
    )
    if draft_model.markov_head is not None or draft_model.confidence_head is not None:
        raise RuntimeError("LAGUNA_DFLASH_CONFORMANCE_HEADS_PRESENT")
    tokenizer = AutoTokenizer.from_pretrained(
        target_path, trust_remote_code=True, fix_mistral_regex=True
    )
    input_ids = encode_chat_messages(
        tokenizer,
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        enable_thinking=True,
    ).to("cuda")
    captured, handles = _capture_target_layers(
        target_model, [int(value) for value in draft_model.target_layer_ids]
    )
    trace_path = output_root / "trace.json"
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
    ) as profiler:
        with record_function("deepspec_dflash_target_prefill"):
            target_output = target_model(
                input_ids=input_ids,
                position_ids=torch.arange(input_ids.shape[1], device="cuda").unsqueeze(
                    0
                ),
                past_key_values=DynamicCache(),
                use_cache=True,
                output_hidden_states=True,
                logits_to_keep=1,
            )
            anchor = target_output.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        missing = set(draft_model.target_layer_ids) - set(captured)
        if missing:
            raise RuntimeError(f"LAGUNA_DFLASH_TARGET_HOOKS_MISSING:{sorted(missing)}")
        raw_target_features = torch.cat(
            [captured[int(layer_id)] for layer_id in draft_model.target_layer_ids],
            dim=-1,
        )
        combined = _combined_target_feature(draft_model, raw_target_features)
        proposal_length = int(draft_model.proposal_length)
        draft_input_ids = torch.full(
            (1, proposal_length),
            int(draft_model.mask_token_id),
            dtype=torch.long,
            device="cuda",
        )
        draft_input_ids[:, 0] = anchor[:, 0]
        position_ids = torch.arange(
            input_ids.shape[1] + proposal_length, device="cuda"
        ).unsqueeze(0)
        with record_function("deepspec_dflash_backbone"):
            hidden = forward_dspark_draft_block(
                draft_model,
                draft_input_ids=draft_input_ids,
                position_ids=position_ids,
                past_key_values_draft=DynamicCache(),
                target_hidden_states=raw_target_features,
                start=input_ids.shape[1],
                block_size=proposal_length,
            )
        logits = draft_model.compute_logits(hidden)
        proposal_token_ids = logits.argmax(dim=-1)
        torch.cuda.synchronize()
    profiler.export_chrome_trace(str(trace_path))
    for handle in handles:
        handle.remove()
    boundaries = {
        "prompt_token_ids": _save_tensor(output_root, "prompt_token_ids", input_ids),
        "raw_target_features": _save_tensor(
            output_root, "raw_target_features", raw_target_features
        ),
        "combined_target_feature": _save_tensor(
            output_root, "combined_target_feature", combined.squeeze(0)
        ),
        "draft_backbone_hidden_state": _save_tensor(
            output_root, "draft_backbone_hidden_state", hidden.squeeze(0)
        ),
        "base_logits": _save_tensor(output_root, "base_logits", logits.squeeze(0)),
        "proposal_token_ids": _save_tensor(
            output_root, "proposal_token_ids", proposal_token_ids.squeeze(0)
        ),
    }
    manifest: dict[str, object] = {
        "schema_version": 1,
        "implementation": "respectmathias_deepspec_dflash",
        "prompt": prompt,
        "prompt_token_ids": input_ids.cpu().tolist()[0],
        "target_revision": TARGET_REVISION,
        "draft_checkpoint_sha256": _sha256(draft_path / "model.safetensors"),
        "boundaries": boundaries,
        "trace": {
            "path": trace_path.name,
            "sha256": _sha256(trace_path),
            "bytes": trace_path.stat().st_size,
        },
    }
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--target-path", type=Path, required=True)
    parser.add_argument("--draft-path", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run_capture(**vars(args)), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
