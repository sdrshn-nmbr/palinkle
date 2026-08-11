import jax
import jax.numpy as jnp
import jax.experimental.pallas as pl
import jax.experimental.pallas.tpu as pltpu

def workload(query, key, value):
    B, H, S, D = query.shape
    
    def retention_kernel(q_ref, k_ref, v_ref, out_ref):
        b = pl.program_id(0)
        h = pl.program_id(1)
        
        q = q_ref[...].astype(jnp.float32)  # (1, 1, S, D) -> squeeze
        k = k_ref[...].astype(jnp.float32)
        v = v_ref[...].astype(jnp.float32)
        
        # Squeeze batch/head dims
        q = q.reshape(S, D)
        k = k.reshape(S, D)
        v = v.reshape(S, D)
        
        gamma = 1.0 - jnp.exp2(-5.0 - (float(H) - float(h)))
        
        positions = jnp.arange(S, dtype=jnp.float32)
        distance = positions[:, None] - positions[None, :]  # (S, S)
        
        causal_mask = (distance >= 0).astype(jnp.float32)
        
        log_gamma = jnp.log(jnp.array([gamma], dtype=jnp.float32))
        decay = jnp.exp(log_gamma * jnp.maximum(distance, 0.0))
        decay = decay * causal_mask
        
        qk = jnp.dot(q, k.T)  # (S, S)
        qk = qk * decay
        
        retention_sum = jnp.sum(jnp.abs(qk), axis=-1, keepdims=True)
        retention_sum = jnp.maximum(retention_sum, 1.0)
        
        qk = qk / retention_sum
        
        out = jnp.dot(qk, v)  # (S, D)
        
        # Reshape back to (1, 1, S, D)
        out_ref[...] = out.reshape(1, 1, S, D).astype(query.dtype)
    
    grid = (B, H)
    out_shape = jax.ShapeDtypeStruct(query.shape, query.dtype)
    
    return pl.pallas_call(
        retention_kernel,
        out_shape=out_shape,
        grid=grid,
        in_specs=(
            pl.BlockSpec((1, 1, S, D), lambda b, h: (b, h, 0, 0)),
            pl.BlockSpec((1, 1, S, D), lambda b, h: (b, h, 0, 0)),
            pl.BlockSpec((1, 1, S, D), lambda b, h: (b, h, 0, 0)),
        ),
        out_specs=pl.BlockSpec((1, 1, S, D), lambda b, h: (b, h, 0, 0)),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel", "parallel")),
    )(query, key, value)
