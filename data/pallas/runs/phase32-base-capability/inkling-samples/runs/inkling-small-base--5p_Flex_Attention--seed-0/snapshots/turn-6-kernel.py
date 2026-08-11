import jax
import jax.numpy as jnp
import jax.experimental.pallas as pl
import jax.experimental.pallas.tpu as pltpu

def workload(q, k, v, rel_pos_bias):
    batch, heads, seq_len, head_dim = q.shape
    
    def kernel(q_ref, k_ref, v_ref, rel_ref, out_ref):
        # Load full q, k, v for this batch/head
        q_local = q_ref[...]
        k_local = k_ref[...]
        v_local = v_ref[...]
        
        # Compute attention scores
        sm_scale = head_dim ** -0.5
        attn = jnp.einsum('qd,kd->qk', q_local, k_local) * sm_scale
        
        # Add relative position bias (for this head)
        # rel_ref is [seq, seq] for this head
        rel_local = rel_ref[...]
        attn = attn + rel_local
        
        # Causal mask
        causal = jnp.tril(jnp.ones((seq_len, seq_len), dtype=jnp.bool_))
        attn = jnp.where(causal, attn, -1e30)
        
        # Softmax over key axis (last axis)
        attn = jax.nn.softmax(attn, axis=-1)
        
        # Apply to v
        out_local = jnp.einsum('qk,kd->qd', attn, v_local)
        out_ref[...] = out_local
    
    # Grid over batch and heads
    grid = (batch, heads)
    
    # Block specs: each instance gets full sequence for one batch/head
    in_specs = (
        pl.BlockSpec((1, 1, seq_len, head_dim), lambda b, h: (b, h, 0, 0)),
        pl.BlockSpec((1, 1, seq_len, head_dim), lambda b, h: (b, h, 0, 0)),
        pl.BlockSpec((1, 1, seq_len, head_dim), lambda b, h: (b, h, 0, 0)),
        pl.BlockSpec((1, seq_len, seq_len), lambda b, h: (h, 0, 0)),
    )
    out_specs = pl.BlockSpec((1, 1, seq_len, head_dim), lambda b, h: (b, h, 0, 0))
    
    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct(q.shape, q.dtype),
        grid=grid,
        in_specs=in_specs,
        out_specs=out_specs,
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel", "parallel")),
    )(q, k, v, rel_pos_bias)
