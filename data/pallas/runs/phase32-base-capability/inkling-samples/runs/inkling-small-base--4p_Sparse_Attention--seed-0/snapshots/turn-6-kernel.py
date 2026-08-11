import jax
import jax.numpy as jnp
import jax.lax as lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

def workload(q, k, v):
    # Config from instruction
    S = 4096
    H_q = 64
    H_kv = 8
    num_q_per_kv = H_q // H_kv  # 8
    head_dim = 128
    
    # Repeat k and v along head axis to match query heads
    # But we do this inside the kernel via block specs and indexing
    # Actually the semantics require repeating k and v before einsum.
    # We can either do it outside (plain JAX) or inside kernel.
    # To preserve semantics and use Pallas, let's do the repeat inside kernel
    # by loading the appropriate kv head and repeating.
    
    block_h = 8
    block_s = 128
    grid_h = H_q // block_h  # 8
    grid_s = S // block_s    # 32
    
    def sparse_attn_kernel(q_ref, k_ref, v_ref, out_ref):
        # Load blocks
        q_local = q_ref[...]  # (block_h, block_s, head_dim)
        k_local = k_ref[...]  # (1, S, head_dim) - full seq for kv head
        v_local = v_ref[...]  # (1, S, head_dim)
        
        # Repeat k and v to match query heads in block
        k_exp = jnp.repeat(k_local, block_h, axis=0)  # (block_h, S, head_dim)
        v_exp = jnp.repeat(v_local, block_h, axis=0)  # (block_h, S, head_dim)
        
        # Compute attention scores
        attn = jnp.einsum("hqd,hkd->hqk", q_local, k_exp)  # (block_h, block_s, S)
        
        # Apply causal mask
        # Query positions: s_start + i where s_start = pl.program_id(1) * block_s
        s_start = pl.program_id(1) * block_s
        # Create position arrays
        q_pos = jnp.arange(block_s) + s_start  # (block_s,)
        k_pos = jnp.arange(S)  # (S,)
        # Causal mask: k_pos <= q_pos[:, None]
        causal_mask = k_pos[None, :] <= q_pos[:, None]  # (block_s, S)
        # Broadcast to (block_h, block_s, S)
        causal_mask = jnp.broadcast_to(causal_mask[None, :, :], (block_h, block_s, S))
        
        attn = jnp.where(causal_mask, attn, jnp.array(-1e30, dtype=attn.dtype))
        
        # Softmax over last axis
        attn = jax.nn.softmax(attn, axis=-1)
        
        # Compute output
        out_local = jnp.einsum("hqk,hkd->hqd", attn, v_exp)  # (block_h, block_s, head_dim)
        
        out_ref[...] = out_local
    
    out_shape = jax.ShapeDtypeStruct(q.shape, q.dtype)
    
    return pl.pallas_call(
        sparse_attn_kernel,
        out_shape=out_shape,
        grid=(grid_h, grid_s),
        in_specs=(
            pl.BlockSpec((block_h, block_s, head_dim), lambda h, s: (h * block_h, s * block_s, 0)),
            pl.BlockSpec((1, S, head_dim), lambda h, s: (h, 0, 0)),
            pl.BlockSpec((1, S, head_dim), lambda h, s: (h, 0, 0)),
        ),
        out_specs=pl.BlockSpec((block_h, block_s, head_dim), lambda h, s: (h * block_h, s * block_s, 0)),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel", "parallel")),
    )(q, k, v)
