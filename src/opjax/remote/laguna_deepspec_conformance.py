"""Capture one original DeepSpec Laguna DSpark proposal round."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

from huggingface_hub import snapshot_download
import numpy as np
import torch
from torch.profiler import ProfilerActivity, profile, record_function
from transformers import AutoTokenizer, DynamicCache

from deepspec.data.parser import encode_chat_messages
from deepspec.eval.dspark.draft_ops import forward_dspark_draft_block
from deepspec.modeling.dspark.laguna import LagunaDSparkModel
from deepspec.modeling.dspark.laguna import load_laguna_target_model_strict
from deepspec.modeling.dspark.qwen3.modeling import apply_rotary_pos_emb

from opjax.pallas.laguna_speculative import (
    DSPARK_ID,
    DSPARK_REVISION,
    TARGET_ID,
    TARGET_REVISION,
    canonical_sha256,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _save_tensor(root: Path, name: str, value: torch.Tensor) -> dict[str, object]:
    path = root / f"{name}.npy"
    tensor = value.detach().contiguous().cpu()
    if tensor.dtype == torch.bfloat16:
        tensor = tensor.float()
    with path.open("wb") as handle:
        np.save(handle, tensor.numpy(), allow_pickle=False)
    return {
        "path": path.name,
        "sha256": _sha256(path),
        "shape": list(tensor.shape),
        "dtype": str(tensor.numpy().dtype),
        "source_dtype": str(value.dtype),
    }


def _combined_target_feature(
    model: LagunaDSparkModel, raw_target_features: torch.Tensor
) -> torch.Tensor:
    chunks = raw_target_features.split(model.config.hidden_size, dim=-1)
    normalized = [
        norm(chunk) for norm, chunk in zip(model.aux_hidden_norms, chunks, strict=True)
    ]
    return model.hidden_norm(model.fc(torch.cat(normalized, dim=-1)))


def _stage_snapshot(*, repo_id: str, revision: str, destination: Path) -> Path:
    source = Path(
        snapshot_download(
            repo_id=repo_id,
            revision=revision,
            local_files_only=True,
        )
    )
    shutil.copytree(source, destination, symlinks=False)
    return destination


def _capture_target_layers(
    target_model: object, layer_ids: list[int]
) -> tuple[dict[int, torch.Tensor], list[object]]:
    captured: dict[int, torch.Tensor] = {}
    handles: list[object] = []
    backbone = getattr(target_model, "model", target_model)
    for raw_layer_id in layer_ids:
        layer_id = int(raw_layer_id)

        def capture(_module, _inputs, output, *, layer_id: int = layer_id) -> None:
            value = output[0] if isinstance(output, (tuple, list)) else output
            captured[layer_id] = value.detach()

        handles.append(backbone.layers[layer_id].register_forward_hook(capture))
    return captured, handles


def _capture_draft_layers(
    draft_model: LagunaDSparkModel,
) -> tuple[dict[int, torch.Tensor], list[object]]:
    captured: dict[int, torch.Tensor] = {}
    handles: list[object] = []
    for layer_id, layer in enumerate(draft_model.layers):

        def capture(_module, _inputs, output, *, layer_id: int = layer_id) -> None:
            value = output[0] if isinstance(output, (tuple, list)) else output
            captured[layer_id] = value.detach()

        handles.append(layer.register_forward_hook(capture))
    return captured, handles


def _capture_first_layer_operations(
    draft_model: LagunaDSparkModel,
) -> tuple[dict[str, list[torch.Tensor]], list[object]]:
    captured: dict[str, list[torch.Tensor]] = {}
    handles: list[object] = []
    layer = draft_model.layers[0]
    modules = {
        "layer0_input_norm": layer.input_layernorm,
        "layer0_qkv_projection": layer.self_attn.qkv_proj,
        "layer0_q_norm": layer.self_attn.q_norm,
        "layer0_k_norm": layer.self_attn.k_norm,
        "layer0_gate_projection": layer.self_attn.g_proj,
        "layer0_attention_output": layer.self_attn,
        "layer0_post_attention_norm": layer.post_attention_layernorm,
        "layer0_mlp_output": layer.mlp,
    }
    for name, module in modules.items():

        def capture(_module, _inputs, output, *, name: str = name) -> None:
            value = output[0] if isinstance(output, (tuple, list)) else output
            captured.setdefault(name, []).append(value.detach())

        handles.append(module.register_forward_hook(capture))

    def capture_gated_attention(_module, inputs) -> None:
        captured.setdefault("layer0_gated_attention", []).append(inputs[0].detach())

    handles.append(
        layer.self_attn.o_proj.register_forward_pre_hook(capture_gated_attention)
    )
    return captured, handles


@torch.inference_mode()
def run_capture(
    *,
    output_root: Path,
    prompt: str,
    target_path: Path | None = None,
    draft_path: Path | None = None,
) -> dict[str, object]:
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    output_root.mkdir(parents=True, exist_ok=False)
    staging_root = Path("/tmp/opjax-conformance-models")
    target_path = target_path or _stage_snapshot(
        repo_id=TARGET_ID,
        revision=TARGET_REVISION,
        destination=staging_root / "target",
    )
    draft_path = draft_path or _stage_snapshot(
        repo_id=DSPARK_ID,
        revision=DSPARK_REVISION,
        destination=staging_root / "draft",
    )
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
    tokenizer = AutoTokenizer.from_pretrained(
        target_path,
        trust_remote_code=True,
        fix_mistral_regex=True,
    )
    input_ids = encode_chat_messages(
        tokenizer,
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        enable_thinking=True,
    ).to("cuda")
    captured, target_handles = _capture_target_layers(
        target_model, [int(value) for value in draft_model.target_layer_ids]
    )
    draft_layers, draft_handles = _capture_draft_layers(draft_model)
    draft_operations, operation_handles = _capture_first_layer_operations(draft_model)
    trace_path = output_root / "trace.json"
    activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
    with profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    ) as profiler:
        with record_function("deepspec_target_prefill"):
            target_cache = DynamicCache()
            target_output = target_model(
                input_ids=input_ids,
                position_ids=torch.arange(input_ids.shape[1], device="cuda").unsqueeze(
                    0
                ),
                past_key_values=target_cache,
                use_cache=True,
                output_hidden_states=True,
                logits_to_keep=1,
            )
            anchor = torch.argmax(target_output.logits[:, -1, :], dim=-1, keepdim=True)
        missing = set(draft_model.target_layer_ids) - set(captured)
        if missing:
            raise RuntimeError(f"DEEPSPEC_TARGET_HOOKS_MISSING:{sorted(missing)}")
        raw_target_features = torch.cat(
            [captured[int(layer_id)] for layer_id in draft_model.target_layer_ids],
            dim=-1,
        )
        with record_function("deepspec_combine_target_features"):
            combined_target_feature = _combined_target_feature(
                draft_model, raw_target_features
            )
        block_size = int(draft_model.block_size)
        proposal_length = int(draft_model.proposal_length)
        draft_input_ids = torch.full(
            (1, block_size),
            int(draft_model.mask_token_id),
            dtype=torch.long,
            device="cuda",
        )
        draft_input_ids[:, 0] = anchor[:, 0]
        position_ids = torch.arange(
            input_ids.shape[1] + block_size + 1, device="cuda"
        ).unsqueeze(0)
        draft_positions = position_ids[
            :, input_ids.shape[1] : input_ids.shape[1] + block_size
        ]
        draft_input_embeddings = draft_model.embed_tokens(draft_input_ids)
        with record_function("deepspec_draft_backbone"):
            block_hidden = forward_dspark_draft_block(
                draft_model,
                draft_input_ids=draft_input_ids,
                position_ids=position_ids,
                past_key_values_draft=DynamicCache(),
                target_hidden_states=raw_target_features,
                start=input_ids.shape[1],
                block_size=block_size,
            )
        with record_function("deepspec_draft_backbone_exact_proposal_width"):
            exact_width_hidden = forward_dspark_draft_block(
                draft_model,
                draft_input_ids=draft_input_ids[:, :proposal_length],
                position_ids=position_ids[:, :-1],
                past_key_values_draft=DynamicCache(),
                target_hidden_states=raw_target_features,
                start=input_ids.shape[1],
                block_size=proposal_length,
            )
        proposal_hidden = block_hidden[:, :proposal_length, :]
        with record_function("deepspec_base_logits"):
            base_logits = draft_model.compute_logits(proposal_hidden)
        biases: list[torch.Tensor] = []
        corrected: list[torch.Tensor] = []
        tokens: list[torch.Tensor] = []
        embeddings: list[torch.Tensor] = []
        previous = anchor[:, 0]
        with record_function("deepspec_markov_chain"):
            for step in range(proposal_length):
                embedding = draft_model.markov_head.get_prev_embeddings(previous)
                bias = draft_model.markov_head.project_bias(embedding)
                logits = base_logits[:, step, :] + bias
                token = torch.argmax(logits, dim=-1)
                embeddings.append(embedding)
                biases.append(bias)
                corrected.append(logits)
                tokens.append(token)
                previous = token
        proposal_token_ids = torch.stack(tokens, dim=1)
        previous_token_ids = torch.cat([anchor, proposal_token_ids[:, :-1]], dim=1)
        with record_function("deepspec_confidence"):
            confidence_logits = draft_model.predict_confidence_step(
                proposal_hidden,
                prev_token_ids=previous_token_ids,
            )
        if confidence_logits is None:
            raise RuntimeError("DEEPSPEC_CONFIDENCE_HEAD_MISSING")
        torch.cuda.synchronize()
    profiler.export_chrome_trace(str(trace_path))
    for handle in [*target_handles, *draft_handles, *operation_handles]:
        handle.remove()

    missing_draft_layers = set(range(len(draft_model.layers))) - set(draft_layers)
    if missing_draft_layers:
        raise RuntimeError(
            f"DEEPSPEC_DRAFT_HOOKS_MISSING:{sorted(missing_draft_layers)}"
        )
    query_qkv = draft_operations["layer0_qkv_projection"][0]
    context_qkv = draft_operations["layer0_qkv_projection"][1]
    q_width = draft_model.layers[0].self_attn.q_width
    kv_width = draft_model.layers[0].self_attn.k_width
    query_q, query_k, query_v = torch.split(
        query_qkv, [q_width, kv_width, kv_width], dim=-1
    )
    _context_q, context_k, context_v = torch.split(
        context_qkv, [q_width, kv_width, kv_width], dim=-1
    )
    num_q_heads = int(draft_model.config.num_attention_heads)
    num_kv_heads = int(draft_model.config.num_key_value_heads)
    head_dim = int(draft_model.config.head_dim)
    query_q = (
        draft_model.layers[0]
        .self_attn.q_norm(query_q.view(1, block_size, num_q_heads, head_dim))
        .transpose(1, 2)
    )
    all_k = torch.cat([context_k, query_k], dim=1).view(
        1, input_ids.shape[1] + block_size, num_kv_heads, head_dim
    )
    all_v = torch.cat([context_v, query_v], dim=1).view(
        1, input_ids.shape[1] + block_size, num_kv_heads, head_dim
    )
    all_k = draft_model.layers[0].self_attn.k_norm(all_k).transpose(1, 2)
    all_k_before_rope = all_k
    all_v = all_v.transpose(1, 2)
    position_embeddings = draft_model.rotary_emb(
        draft_input_embeddings,
        position_ids[:, : input_ids.shape[1] + block_size],
    )
    query_q, all_k = apply_rotary_pos_emb(query_q, all_k, *position_embeddings)

    stacked_bias = torch.stack(biases, dim=1)
    stacked_corrected = torch.stack(corrected, dim=1)
    stacked_embeddings = torch.stack(embeddings, dim=1)
    swapped_biases: list[torch.Tensor] = []
    swapped_tokens: list[torch.Tensor] = []
    previous = anchor[:, 0]
    for step in range(proposal_length):
        swapped_embedding = torch.nn.functional.embedding(
            previous, draft_model.markov_head.markov_w2.weight
        )
        swapped_bias = torch.nn.functional.linear(
            swapped_embedding, draft_model.markov_head.markov_w1.weight
        )
        swapped_biases.append(swapped_bias)
        previous = torch.argmax(base_logits[:, step, :] + swapped_bias, dim=-1)
        swapped_tokens.append(previous)
    mutated_bias = torch.stack(swapped_biases, dim=1)
    mutated_tokens = torch.stack(swapped_tokens, dim=1)
    mutation_detected = not torch.equal(proposal_token_ids, mutated_tokens)
    mutation_detected = mutation_detected or not torch.allclose(
        stacked_bias.float(), mutated_bias.float(), rtol=0.05, atol=0.001
    )

    tensors = {
        "prompt_token_ids": _save_tensor(output_root, "prompt_token_ids", input_ids),
        "anchor_token_id": _save_tensor(output_root, "anchor_token_id", anchor),
        "raw_target_features": _save_tensor(
            output_root, "raw_target_features", raw_target_features
        ),
        "combined_target_feature": _save_tensor(
            output_root, "combined_target_feature", combined_target_feature.squeeze(0)
        ),
        "draft_input_ids": _save_tensor(
            output_root, "draft_input_ids", draft_input_ids
        ),
        "draft_positions": _save_tensor(
            output_root, "draft_positions", draft_positions
        ),
        "draft_input_embeddings": _save_tensor(
            output_root, "draft_input_embeddings", draft_input_embeddings
        ),
        "layer0_query_q_after_rope": _save_tensor(
            output_root,
            "layer0_query_q_after_rope",
            query_q.transpose(1, 2),
        ),
        "layer0_query_k_after_rope": _save_tensor(
            output_root,
            "layer0_query_k_after_rope",
            all_k[:, :, input_ids.shape[1] :, :].transpose(1, 2),
        ),
        "layer0_query_v": _save_tensor(
            output_root,
            "layer0_query_v",
            all_v[:, :, input_ids.shape[1] :, :].transpose(1, 2),
        ),
        "layer0_context_k_before_rope": _save_tensor(
            output_root,
            "layer0_context_k_before_rope",
            all_k_before_rope[:, :, : input_ids.shape[1], :].transpose(1, 2),
        ),
        "layer0_context_k_after_rope": _save_tensor(
            output_root,
            "layer0_context_k_after_rope",
            all_k[:, :, : input_ids.shape[1], :].transpose(1, 2),
        ),
        "layer0_context_v": _save_tensor(
            output_root,
            "layer0_context_v",
            all_v[:, :, : input_ids.shape[1], :].transpose(1, 2),
        ),
        **{
            f"draft_layer_{layer_id}_output": _save_tensor(
                output_root,
                f"draft_layer_{layer_id}_output",
                draft_layers[layer_id],
            )
            for layer_id in sorted(draft_layers)
        },
        **{
            f"{name}_{index}": _save_tensor(
                output_root,
                f"{name}_{index}",
                value,
            )
            for name, values in sorted(draft_operations.items())
            for index, value in enumerate(values)
        },
        "draft_backbone_hidden_state": _save_tensor(
            output_root, "draft_backbone_hidden_state", proposal_hidden.squeeze(0)
        ),
        "draft_backbone_hidden_state_exact_width": _save_tensor(
            output_root,
            "draft_backbone_hidden_state_exact_width",
            exact_width_hidden.squeeze(0),
        ),
        "base_logits": _save_tensor(output_root, "base_logits", base_logits.squeeze(0)),
        "markov_embedding": _save_tensor(
            output_root, "markov_embedding", stacked_embeddings.squeeze(0)
        ),
        "markov_bias": _save_tensor(
            output_root, "markov_bias", stacked_bias.squeeze(0)
        ),
        "corrected_logits": _save_tensor(
            output_root, "corrected_logits", stacked_corrected.squeeze(0)
        ),
        "confidence_logits": _save_tensor(
            output_root, "confidence_logits", confidence_logits.squeeze(0)
        ),
        "proposal_token_ids": _save_tensor(
            output_root, "proposal_token_ids", proposal_token_ids.squeeze(0)
        ),
    }
    manifest: dict[str, object] = {
        "schema_version": 1,
        "implementation": "respectmathias_deepspec",
        "provenance": {
            "revision": "787db11ea347ac3944233e5aa9c7f1bd8a9b5ced",
            "target_revision": TARGET_REVISION,
            "draft_revision": (
                DSPARK_REVISION
                if str(draft_path).startswith(str(staging_root))
                else _sha256(draft_path / "model.safetensors")
            ),
            "source_sha256": _sha256(Path(__file__)),
        },
        "prompt": prompt,
        "prompt_token_ids": input_ids.cpu().tolist()[0],
        "boundaries": tensors,
        "trace": {
            "path": trace_path.name,
            "sha256": _sha256(trace_path),
            "bytes": trace_path.stat().st_size,
        },
        "mutation_controls": {
            "markov_matrix_swap": {
                "detected": mutation_detected,
                "proposal_tokens_changed": not torch.equal(
                    proposal_token_ids, mutated_tokens
                ),
                "max_abs_bias_delta": float(
                    torch.max(
                        torch.abs(stacked_bias.float() - mutated_bias.float())
                    ).item()
                ),
            }
        },
    }
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    del target_model, draft_model
    torch.cuda.empty_cache()
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--target-path", type=Path)
    parser.add_argument("--draft-path", type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            run_capture(output_root=args.output_root, prompt=args.prompt),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
