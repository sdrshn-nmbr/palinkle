from __future__ import annotations

import torch

from vllm.model_executor.models.laguna import LagunaForCausalLM

from opjax.remote.laguna_dspark_capture import (
    begin_target_capture_round,
    capture_target_tensor,
)


class CapturedLagunaForCausalLM(LagunaForCausalLM):
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors=None,
        inputs_embeds: torch.Tensor | None = None,
    ):
        output = super().forward(
            input_ids,
            positions,
            intermediate_tensors,
            inputs_embeds,
        )
        if not isinstance(output, tuple) or len(output) != 2:
            raise RuntimeError("LAGUNA_TARGET_CAPTURE_AUXILIARY_OUTPUT_MISSING")
        hidden_states, auxiliary = output
        if len(auxiliary) != 5:
            raise RuntimeError(
                f"LAGUNA_TARGET_CAPTURE_AUXILIARY_COUNT:{len(auxiliary)}"
            )
        begin_target_capture_round()
        capture_target_tensor("target_input_ids", input_ids)
        capture_target_tensor("target_positions", positions)
        capture_target_tensor("target_last_hidden_states", hidden_states)
        capture_target_tensor(
            "target_aux_hidden_states", torch.cat(auxiliary, dim=-1)
        )
        return output
