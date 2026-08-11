import jax
import jax.numpy as jnp
from jax import lax
import jax.nn as jnn
import jax.experimental.pallas as pl
import jax.experimental.pallas.tpu as pltpu

def workload(x, router_weights, expert_gate_kernels, expert_up_kernels, expert_down_kernels):
    def kernel(x_ref, router_weights_ref, expert_gate_kernels_ref, expert_up_kernels_ref, expert_down_kernels_ref, out_ref):
        x = x_ref[...]
        router_weights = router_weights_ref[...]
        expert_gate_kernels = expert_gate_kernels_ref[...]
        expert_up_kernels = expert_up_kernels_ref[...]
        expert_down_kernels = expert_down_kernels_ref[...]
        
        B, S, E = x.shape
        N = router_weights.shape[-1]
        K = 2  # num_experts_per_tok
        
        logits = jnp.dot(x, router_weights)
        top_k_logits, top_k_indices = lax.top_k(logits, K)
        router_probs = jnn.softmax(top_k_logits, axis=-1)
        
        gate_out = jnn.silu(jnp.einsum("bse,nem->bsnm", x, expert_gate_kernels))
        up_out = jnp.einsum("bse,nem->bsnm", x, expert_up_kernels)
        hidden = gate_out * up_out
        expert_outputs = jnp.einsum("bsnm,nme->bsne", hidden, expert_down_kernels)
        
        one_hot = jnn.one_hot(top_k_indices, N)
        weighted = one_hot * router_probs[..., None]
        expert_weights = jnp.sum(weighted, axis=2)
        
        output = jnp.einsum("bsne,bsn->bse", expert_outputs, expert_weights)
        out_ref[...] = output.astype(jnp.float32)
    
    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, jnp.float32),
        grid=(),
        in_specs=(
            pl.no_block_spec,
            pl.no_block_spec,
            pl.no_block_spec,
            pl.no_block_spec,
            pl.no_block_spec,
        ),
        out_specs=pl.no_block_spec,
    )(x, router_weights, expert_gate_kernels, expert_up_kernels, expert_down_kernels)
