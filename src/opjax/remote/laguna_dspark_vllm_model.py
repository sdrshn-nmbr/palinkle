"""Laguna-specific DSpark model for vLLM."""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import nn
from vllm.config import VllmConfig
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.models.laguna_dflash import DFlashLagunaModel
from vllm.model_executor.models.qwen3_dspark import (
    DSparkConfidenceHead,
    DSparkMarkovHead,
    Qwen3DSparkForCausalLM,
)
from vllm.model_executor.models.utils import AutoWeightsLoader, maybe_prefix


class LagunaDSparkModel(DFlashLagunaModel):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__(vllm_config=vllm_config, prefix=prefix)
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

    def combine_hidden_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
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
        return result.squeeze(0) if needs_squeeze else result

    def load_weights(
        self, weights: Iterable[tuple[str, torch.Tensor]]
    ) -> set[str]:
        model_weights: dict[str, torch.Tensor] = {}
        skipped_target_tensors: set[str] = set()
        source_tensor_count = 0
        for name, loaded_weight in weights:
            source_tensor_count += 1
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
