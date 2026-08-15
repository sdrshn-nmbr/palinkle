import jax
import jax.numpy as jnp
import jax.lax as lax
import jax.nn as nn
import pallas as pl
import jax.pallas as pl
import jax.pallas.tpu as pltpu

# Configuration
CONFIG = {
    "batch": 2,
    "emb_dim": 4096,
    "mlp_dim": 14336,
    "model": "Mixtral-8x7B",
    "name": "mixtral_8x7b_moe",
    "num_experts": 8,
    "num_experts_per_tok": 2,
    "operator": "sparse_moe",
    "seq_len": 4096
}


def sparse_moe_kernel(
    x_ref,
    router_weights_ref,
    expert_gate_kernels_ref,
    expert_up_kernels_ref,
    expert_down_kernels_ref,
    out_ref,
):
    """Pallas kernel for Sparse MoE computation."""
    # Get shapes from references
    B, S, E = x_ref.shape
    N = router_weights_ref.shape[0]
    K = CONFIG["num_experts_per_tok"]
    M = CONFIG["mlp_dim"]
    
    # Compute logits = x @ router_weights
    # x: [B, S, E], router_weights: [S, N] -> logits: [B, S, N]
    logits = jnp.dot(x_ref, router_weights_ref)
    
    # Get top-k indices and values
    top_k_logits, top_k_indices = lax.top_k(logits, K)
    
    # Compute router probabilities
    router_probs = nn.softmax(top_k_logits, axis=-1)
    
    # Compute gate_out = silu(einsum("bse,nem->bsnm", x, expert_gate_kernels))
    # x: [B, S, E], expert_gate_kernels: [N, E, M] -> [B, S, N, M]
    gate_out = nn.silu(jnp.einsum("bse,nem->bsnm", x_ref, expert_gate_kernels_ref))
    
    # Compute up_out = einsum("bse,nem->bsnm", x, expert_up_kernels)
    # x: [B, S, E], expert_up_kernels: [N, E, M] -> [B, S, N, M]
    up_out = jnp.einsum("bse,nem->bsnm", x_ref, expert_up_kernels_ref)
    
    # Compute hidden = gate_out * up_out
    hidden = gate_out * up_out
    
    # Compute expert_outputs = einsum("bsnm,nme->bsne", hidden, expert_down_kernels)
    # hidden: [B, S, N, M], expert_down_kernels: [N, M, E] -> [B, S, N, E]
    expert_outputs = jnp.einsum("bsnm,nme->bsne", hidden, expert_down_kernels_ref)
    
    # Compute one_hot from top_k_indices
    # top_k_indices: [B, S, K], N -> one_hot: [B, S, K, N]
    one_hot = nn.one_hot(top_k_indices, N)
    
    # Compute weighted = one_hot * router_probs[..., None]
    # one_hot: [B, S, K, N], router_probs: [B, S, K] -> weighted: [B, S, K, N]
    weighted = one_hot * router_probs[..., None]
    
    # Compute expert_weights = sum(weighted, axis=2)
    # weighted: [B, S, K, N] -> expert_weights: [B, S, N]
    expert_weights = jnp.sum(weighted, axis=2)
    
    # Compute output = einsum("bsne,bsn->bse", expert_outputs, expert_weights)
    # expert_outputs: [B, S, N, E], expert_weights: [B, S, N] -> output: [B, S, E]
    output = jnp.einsum("bsne,bsn->bse", expert_outputs, expert_weights)
    
    out_ref[...] = output


def workload(x, router_weights, expert_gate_kernels, expert_up_kernels, expert_down_kernels):
    """Sparse MoE workload using Pallas kernel."""
    B, S, E = x.shape
    N = router_weights.shape[0]
    K = CONFIG["num_experts_per_tok"]
    M = CONFIG["mlp_dim"]
    
    # Define block shapes for TPU
    # Use block sizes that are multiples of 8 for bf16
    block_b = 1  # Batch dimension
    block_s = 128  # Sequence dimension (multiple of 8)
    block_e = 128  # Embedding dimension (multiple of 8)
    
    # Grid specification
    grid = (B, S // block_s, E // block_e)
    
    return pl.pallas_call(
        sparse_moe_kernel,
        out_shape=jax.ShapeDtypeStruct((B, S, E), jnp.float32),
        grid=grid,
        in_specs=(
            pl.BlockSpec((block_b, block_s, block_e), lambda b, s, e: (b, s * block_s, e * block_e)),
            pl.BlockSpec((S, N), lambda b, s, e: (0, 0)),
            pl.BlockSpec((N, E, M), lambda b, s, e: (0, 0, 0)),
            pl.BlockSpec((N, E, M), lambda b, s, e: (0, 0, 0)),
            pl.BlockSpec((N, M, E), lambda b, s, e: (0, 0, 0)),
        ),
        out_specs=pl.BlockSpec((block_b, block_s, block_e), lambda b, s, e: (b, s * block_s, e * block_e)),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel", "parallel")
        ),
    )(x, router_weights, expert_gate_kernels, expert_up_kernels, expert_down_kernels)
