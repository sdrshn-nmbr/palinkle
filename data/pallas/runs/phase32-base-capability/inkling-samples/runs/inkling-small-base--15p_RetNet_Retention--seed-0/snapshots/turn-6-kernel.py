import jax
import jax.numpy as jnp
import jax.experimental.pallas as pl
import jax.experimental.pallas.tpu as pltpu

def workload(query, key, value):
    B, H, S, D = query.shape
    
    def retention_kernel(q_ref, k_ref, v_ref, out_ref):
        q = q_ref[0, 0, :, :]  # (S, D)
        k = k_ref[0, 0, :, :]
        v = v_ref[0, 0, :, :]
        
        h = pl.program_id(1)
        gamma = 1.0 - jnp.exp2(-5.0 - jnp.float32(h))
        
        q_f32 = q.astype(jnp.float32)
        k_f32 = k.astype(jnp.float32)
        qk = jnp.dot(q_f32, k_f32.T)  # (S, S)
        
        positions = jnp.arange(S, dtype=jnp.float32)
        distance = positions[:, None] - positions[None, :]
        causal_mask = (distance >= 0).astype(jnp.float32)
        log_gamma = jnp.log(gamma)
        decay = jnp.exp(log_gamma * jnp.maximum(distance, 0.0))
        decay = decay * causal_mask
        
        qk = qk * decay
        
        retention_sum = jnp.sum(jnp.abs(qk), axis=-1, keepdims=True)
        retention_sum = jnp.maximum(retention_sum, 1.0)
        qk = qk / retention_sum
        
        qk_bf16 = qk.astype(q.dtype)
        out_f32 = jnp.dot(qk_bf16.astype(jnp.float32), v.astype(jnp.float32))
        out_ref[0, 0, :, :] = out_f32.astype(q.dtype)
    
    out_shape = jax.ShapeDtypeStruct(query.shape, query.dtype)
    
    return pl.pallas_call(
        retention_kernel,
        out_shape=out_shape,
        grid=(B, H),
        in_specs=(
            pl.BlockSpec((1, 1, S, D), lambda b, h: (b, h, 0, 0)),
            pl.BlockSpec((1, 1, S, D), lambda b, h: (b, h, 0, 0)),
            pl.BlockSpec((1, 1, S, D), lambda b, h: (b, h, 0, 0)),
        ),
        out_specs=pl.BlockSpec((1, 1, S, D), lambda b, h: (b, h, 0, 0)),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel", "parallel")),
    )(query, key, value)
