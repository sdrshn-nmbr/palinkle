from __future__ import annotations

import torch
from vllm.model_executor.models.qwen3_dflash import DFlashQwen3ForCausalLM

from opjax.remote.laguna_dspark_capture import (
    capture_tensor,
    load_target_feature_override,
)


class CapturedLagunaDFlashForCausalLM(DFlashQwen3ForCausalLM):
    def combine_hidden_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
        override = load_target_feature_override()
        if override is not None:
            hidden_states = override.to(
                device=hidden_states.device, dtype=hidden_states.dtype
            )
        combined = super().combine_hidden_states(hidden_states)
        capture_tensor("combined_target_feature", combined)
        return combined

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden = super().forward(input_ids, positions, inputs_embeds)
        capture_tensor("draft_backbone_hidden_state", hidden)
        return hidden

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        logits = super().compute_logits(hidden_states)
        if logits is not None:
            capture_tensor("base_logits", logits)
        return logits
