import jax
import jax.numpy as jnp
import jax.experimental.pallas as pl
import jax.experimental.pallas.tpu as pltpu

def workload(q, k, v):
    # GQA: repeat k and v to match query heads
    H_q = q.shape[0]
    H_kv = k.shape[0]
    num_q_per_kv = H_q // H_kv
    k = jnp.repeat(k, num_q_per_kv, axis=0)
    v = jnp.repeat(v, num_q_per_kv, axis=0)
    
    def attn_kernel(q_ref, k_ref, v_ref, out_ref):
        # Block shapes: q [8,128,128], k [8,4096,128], v [8,4096,128], out [8,128,128]
        q_f = q_ref.astype(jnp.float32)
        k_f = k_ref.astype(jnp.float32)
        # Scores: [8, 128, 4096]
        s_scores = jnp.dot(q_f, jnp.swapaxes(k_f, -2, -1))
        
        # Causal mask based on global positions
        s = pl.program_id(1)
        q_start = s * 128
        q_idx = jnp.arange(128) + q_start  # [128]
        k_idx = jnp.arange(4096)  # [4096]
        # mask: k_idx <= q_idx
        mask = k_idx[None, :] <= q_idx[:, None]  # [128, 4096]
        mask = jnp.broadcast_to(mask[None, :, :], s_scores.shape)
        
        s_scores = jnp.where(mask, s_scores, -1e30)
        
        attn = jax.nn.softmax(s_scores, axis=-1)
        
        v_f = v_ref.astype(jnp.float32)
        out_f = jnp.dot(attn, v_f)
        out_ref[...] = out_f.astype(jnp.bfloat16)
    
    return pl.pallas_call(
        attn_kernel,
        out_shape=jax.ShapeDtypeStruct(q.shape, q.dtype),
        grid=(8, 32),
        in_specs=(
            pl.BlockSpec((8, 128, 128), lambda h, s: (h, s, 0)),
            pl.BlockSpec((8, 4096, 128), lambda h, s: (h, 0, 0)),
            pl.BlockSpec((8, 4096, 128), lambda h, s: (h, 0, 0)),
        ),
        out_specs=pl.BlockSpec((8, 128, 128), lambda h, s: (h, s, 0)),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel", "parallel")),
    )(q, k, v)
