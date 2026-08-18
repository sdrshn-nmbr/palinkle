"""Laguna-specific DSpark model for vLLM."""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import nn
from vllm.config import VllmConfig
from vllm.model_executor.layers.attention.attention import get_attention_context
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.models.laguna_dflash import DFlashLagunaModel
from vllm.model_executor.models.qwen3_dspark import (
    DSparkConfidenceHead,
    DSparkMarkovHead,
    Qwen3DSparkForCausalLM,
)
from vllm.model_executor.models.utils import AutoWeightsLoader, maybe_prefix

from opjax.remote.laguna_dspark_capture import (
    begin_capture_round,
    capture_step,
    capture_is_active,
    capture_is_configured,
    capture_static_metadata,
    capture_static_tensor,
    capture_tensor,
    logical_context_kv,
    load_target_feature_override,
    validate_single_request_attention_layout,
)


class LagunaDSparkModel(DFlashLagunaModel):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        self._capture_enabled = capture_is_configured()
        config = self.config
        causal = bool(config.dflash_config["causal"])
        for layer in self.layers:
            layer.self_attn.causal = causal
        self.markov_head = DSparkMarkovHead(
            config.vocab_size,
            config.draft_vocab_size,
            config.markov_rank,
            prefix=maybe_prefix(prefix, "markov_head"),
            quant_config=self.quant_config,
        )
        self.confidence_head: DSparkConfidenceHead | None = None
        if config.enable_confidence_head:
            input_dim = config.hidden_size
            if config.confidence_head_with_markov:
                input_dim += config.markov_rank
            self.confidence_head = DSparkConfidenceHead(
                input_dim,
                prefix=maybe_prefix(prefix, "confidence_head"),
                bias=True,
                with_markov=config.confidence_head_with_markov,
            )
        if self._capture_enabled:
            self._register_capture_hooks()
            capture_static_metadata(
                "attention-runtime",
                {
                    "layers": [
                        self._attention_runtime_metadata(layer_id, layer)
                        for layer_id, layer in enumerate(self.layers)
                    ]
                },
            )

    @staticmethod
    def _attention_runtime_metadata(layer_id: int, layer: nn.Module) -> dict[str, object]:
        attention = layer.self_attn.attn
        implementation = attention.impl
        backend = attention.get_attn_backend()
        return {
            "layer": layer_id,
            "backend_name": backend.get_name(),
            "backend_class": f"{backend.__module__}.{backend.__qualname__}",
            "implementation_class": (
                f"{type(implementation).__module__}."
                f"{type(implementation).__qualname__}"
            ),
            "wrapper_sliding_window": attention.sliding_window,
            "implementation_sliding_window": getattr(
                implementation, "sliding_window", None
            ),
            "outer_sliding_window": layer.self_attn.sliding_window,
        }

    def _register_capture_hooks(self) -> None:
        for layer_id, layer in enumerate(self.layers):
            layer.register_forward_hook(self._capture_layer_output(layer_id))
        layer = self.layers[0]
        for name, module in {
            "layer0_input_norm": layer.input_layernorm,
            "layer0_qkv_projection": layer.self_attn.qkv_proj,
            "layer0_q_norm": layer.self_attn.q_norm,
            "layer0_k_norm": layer.self_attn.k_norm,
            "layer0_gate_projection": layer.self_attn.g_proj,
            "layer0_attention_output": layer.self_attn,
            "layer0_post_attention_norm": layer.post_attention_layernorm,
            "layer0_mlp_output": layer.mlp,
        }.items():
            module.register_forward_hook(self._capture_operation(name))
        layer.self_attn.attn.register_forward_pre_hook(
            self._capture_attention_inputs
        )
        layer.self_attn.attn.register_forward_hook(
            self._capture_operation("layer0_raw_attention_output")
        )
        layer.self_attn.o_proj.register_forward_pre_hook(
            self._capture_gated_attention
        )

    @staticmethod
    def _capture_layer_output(layer_id: int):
        def capture(_module, _inputs, output) -> None:
            if not isinstance(output, tuple) or len(output) != 2:
                raise RuntimeError(
                    f"LAGUNA_DSPARK_LAYER_OUTPUT_INVALID:{layer_id}"
                )
            hidden_states, residual = output
            capture_tensor(
                f"draft_layer_{layer_id}_output",
                hidden_states if residual is None else hidden_states + residual,
            )

        return capture

    @staticmethod
    def _capture_operation(name: str):
        def capture(_module, _inputs, output) -> None:
            value = output[0] if isinstance(output, (tuple, list)) else output
            capture_tensor(name, value)

        return capture

    @staticmethod
    def _capture_attention_inputs(module, inputs) -> None:
        if len(inputs) < 3:
            raise RuntimeError("LAGUNA_DSPARK_ATTENTION_INPUTS_INVALID")
        capture_tensor("layer0_query_q_after_rope", inputs[0])
        capture_tensor("layer0_query_k_after_rope", inputs[1])
        capture_tensor("layer0_query_v", inputs[2])
        if not capture_is_active():
            return
        metadata, _attention, kv_cache, slot_mapping = get_attention_context(
            module.layer_name
        )
        for name in ("query_start_loc", "seq_lens", "block_table", "slot_mapping"):
            value = slot_mapping if name == "slot_mapping" else getattr(metadata, name)
            if value is None:
                raise RuntimeError(f"LAGUNA_DSPARK_ATTENTION_METADATA_MISSING:{name}")
            capture_tensor(f"layer0_metadata_{name}", value)
        sequence_lengths = metadata.seq_lens.reshape(-1)
        block_tables = metadata.block_table
        query_length = int(inputs[0].shape[0])
        validate_single_request_attention_layout(
            metadata.query_start_loc,
            sequence_lengths,
            block_tables,
            slot_mapping,
            query_length=query_length,
            block_size=int(kv_cache.shape[2]),
        )
        if kv_cache.shape[-1] != 2 * module.head_size:
            raise RuntimeError(
                "LAGUNA_DSPARK_ATTENTION_CACHE_WIDTH:"
                f"{kv_cache.shape[-1]}:{module.head_size}"
            )
        context_k, context_v = logical_context_kv(
            kv_cache,
            block_tables[0],
            sequence_length=int(sequence_lengths[0].item()),
            query_length=query_length,
        )
        capture_tensor("layer0_logical_context_k", context_k)
        capture_tensor("layer0_logical_context_v", context_v)

    @staticmethod
    def _capture_gated_attention(_module, inputs) -> None:
        capture_tensor("layer0_gated_attention", inputs[0])

    def _project_context_kv(
        self,
        context_states: torch.Tensor,
        num_ctx: int,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        all_k, all_v = super()._project_context_kv(
            context_states,
            num_ctx,
            num_layers,
            num_kv_heads,
            head_dim,
        )
        if self._capture_enabled:
            capture_tensor("layer0_context_k_raw", all_k[0])
            capture_tensor("layer0_context_v", all_v[0])
        return all_k, all_v

    def _normalize_context_k(self, all_k: torch.Tensor) -> torch.Tensor:
        normalized = super()._normalize_context_k(all_k)
        if self._capture_enabled:
            capture_tensor("layer0_context_k_before_rope", normalized[0])
        return normalized

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self._capture_enabled:
            capture_tensor("draft_input_ids", input_ids)
            capture_tensor("draft_positions", positions)
            effective_input_embeddings = (
                self.embed_input_ids(input_ids)
                if inputs_embeds is None
                else inputs_embeds
            )
            capture_tensor("draft_input_embeddings", effective_input_embeddings)
        result = super().forward(input_ids, positions, inputs_embeds)
        if self._capture_enabled:
            capture_tensor("draft_backbone_hidden_state", result)
        return result


class LagunaDSparkForCausalLM(Qwen3DSparkForCausalLM):
    expected_checkpoint_tensor_count = 64
    expected_shared_tensor_count = 2

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        nn.Module.__init__(self)
        speculative = vllm_config.speculative_config
        if speculative is None:
            raise ValueError("Laguna DSpark requires speculative_config.")
        self.draft_model_config = speculative.draft_model_config
        self.config = self.draft_model_config.hf_config
        self.has_own_embed_tokens = False
        self.has_own_lm_head = False

        target_layer_count = vllm_config.model_config.get_num_layers(
            vllm_config.parallel_config
        )
        self.config.target_layer_count = target_layer_count
        target_vocab_size = vllm_config.model_config.get_vocab_size()
        if self.config.draft_vocab_size != target_vocab_size:
            raise ValueError(
                "Laguna DSpark shares target vocabulary tensors and requires "
                f"equal vocabularies ({self.config.draft_vocab_size} != "
                f"{target_vocab_size})."
            )

        self.model = LagunaDSparkModel(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "model"),
        )
        self.lm_head = ParallelLMHead(
            self.config.draft_vocab_size,
            self.config.hidden_size,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        self.logits_processor = LogitsProcessor(self.config.draft_vocab_size)
        self.draft_id_to_target_id = None
        self._capture_enabled = capture_is_configured()

    def combine_hidden_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if capture_is_active():
            begin_capture_round()
        target_feature_override = (
            load_target_feature_override() if capture_is_active() else None
        )
        if target_feature_override is not None:
            override = target_feature_override.to(
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            )
            if override.shape != hidden_states.shape:
                raise RuntimeError(
                    "LAGUNA_TARGET_FEATURE_OVERRIDE_SHAPE:"
                    f"expected={tuple(hidden_states.shape)}:"
                    f"observed={tuple(override.shape)}"
                )
            hidden_states = override
        if self._capture_enabled:
            capture_tensor("raw_target_features", hidden_states)
        needs_squeeze = hidden_states.dim() == 1
        if needs_squeeze:
            hidden_states = hidden_states.unsqueeze(0)
        num_slices = self.model.num_aux_slices
        slice_size = hidden_states.shape[-1] // num_slices
        slices = hidden_states.view(hidden_states.shape[0], num_slices, slice_size)
        normalized = torch.empty_like(slices)
        for index, norm in enumerate(self.model.aux_hidden_norms):
            normalized[:, index, :] = norm(slices[:, index, :])
        combined = normalized.reshape(hidden_states.shape[0], -1)
        result = self.model.hidden_norm(self.model.fc(combined))
        if self._capture_enabled:
            capture_tensor("combined_target_feature", result)
        return result.squeeze(0) if needs_squeeze else result

    def compute_draft_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        active = self._capture_enabled and capture_is_active()
        if active:
            capture_tensor("draft_logits_hidden_state", hidden_states)
        result = super().compute_draft_logits(hidden_states)
        if active:
            capture_tensor("base_logits", result)
            self._opjax_capture_hidden_states = hidden_states
            self._opjax_capture_base_logits = result
            self._opjax_capture_markov_step = 0
        return result

    def markov_embed(self, token_ids: torch.Tensor) -> torch.Tensor:
        if self._capture_enabled:
            capture_tensor("markov_input_token_ids", token_ids)
        result = super().markov_embed(token_ids)
        if self._capture_enabled:
            capture_tensor("markov_embedding", result)
        return result

    def markov_bias(self, markov_embed: torch.Tensor) -> torch.Tensor:
        result = super().markov_bias(markov_embed)
        if self._capture_enabled and capture_is_active():
            capture_tensor("markov_bias", result)
            step = self._opjax_capture_markov_step
            base = capture_step(self._opjax_capture_base_logits, step)
            corrected = base + result
            capture_tensor("corrected_logits_runtime", corrected)
            capture_tensor(
                "proposal_token_ids_runtime", torch.argmax(corrected, dim=-1)
            )
            hidden = capture_step(self._opjax_capture_hidden_states, step)
            if self.model.confidence_head is None:
                raise RuntimeError("LAGUNA_DSPARK_CONFIDENCE_HEAD_MISSING")
            confidence = self.model.confidence_head(hidden, markov_embed)
            capture_tensor("confidence_logits_instrumented", confidence)
            self._opjax_capture_markov_step = step + 1
        return result

    def compute_confidence(
        self, head_hidden: torch.Tensor, markov_embed: torch.Tensor
    ) -> torch.Tensor:
        if self.model.confidence_head is None:
            raise RuntimeError("LAGUNA_DSPARK_CONFIDENCE_HEAD_MISSING")
        logits = self.model.confidence_head(head_hidden, markov_embed)
        if self._capture_enabled:
            capture_tensor("confidence_logits", logits)
        result = torch.sigmoid(logits)
        if self._capture_enabled:
            capture_tensor("confidence_probability", result)
        return result

    def load_weights(
        self, weights: Iterable[tuple[str, torch.Tensor]]
    ) -> set[str]:
        model_weights: dict[str, torch.Tensor] = {}
        skipped_target_tensors: set[str] = set()
        source_tensor_count = 0
        for name, loaded_weight in weights:
            source_tensor_count += 1
            if name in {
                "markov_head.markov_w1.weight",
                "markov_head.markov_w2.weight",
                "confidence_head.proj.weight",
                "confidence_head.proj.bias",
            }:
                capture_static_tensor(name.replace(".", "_"), loaded_weight)
            if "t2d" in name or "d2t" in name:
                continue
            if "embed_tokens" in name or "lm_head" in name:
                skipped_target_tensors.add(name)
                continue
            model_weights[f"model.{name}"] = loaded_weight

        expected_shared = {"embed_tokens.weight", "lm_head.weight"}
        if skipped_target_tensors != expected_shared:
            raise ValueError(
                "Laguna DSpark shared tensor inventory mismatch: "
                f"{sorted(skipped_target_tensors)}"
            )
        if source_tensor_count != self.expected_checkpoint_tensor_count:
            raise ValueError(
                "Laguna DSpark checkpoint tensor count mismatch: "
                f"{source_tensor_count}"
            )

        loader = AutoWeightsLoader(
            self,
            skip_substrs=[
                "mask_embedding",
                "embed_tokens",
                "lm_head",
                "draft_id_to_target_id",
            ],
        )
        loaded = loader.load_weights(model_weights.items())
        if len(loaded) != (
            self.expected_checkpoint_tensor_count - self.expected_shared_tensor_count
        ):
            raise ValueError(
                f"Laguna DSpark loaded tensor count mismatch: {len(loaded)}"
            )
        self.model._build_fused_kv_buffers()
        return loaded
