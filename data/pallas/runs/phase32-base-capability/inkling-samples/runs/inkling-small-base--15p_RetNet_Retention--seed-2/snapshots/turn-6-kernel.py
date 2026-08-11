import jax
import jax.numpy as jnp
import jax.experimental.pallas as pl
import jax.experimental.pallas.tpu as pltpu

def workload(query, key, value):
    B, H, S, D = query.shape
    
    # Precompute gammas, decay, etc. outside kernel since they're constant per head
    # But we need to apply them inside. Let's compute decay inside kernel using scratch.
    
    chunk_s = 256
    grid_s = (S + chunk_s - 1) // chunk_s
    
    def retention_kernel(query_ref, key_ref, value_ref, out_ref):
        # Get program IDs
        b = pl.program_id(0)
        h = pl.program_id(1)
        s_chunk = pl.program_id(2)
        
        s_start = s_chunk * chunk_s
        s_end = jnp.minimum(s_start + chunk_s, S)
        s_size = s_end - s_start
        
        # Load query chunk
        q_chunk = query_ref[b, h, s_start:s_end, :]
        
        # Initialize accumulators
        out_acc = jnp.zeros((s_size, D), dtype=jnp.float32)
        sum_acc = jnp.zeros((s_size, 1), dtype=jnp.float32)
        
        # Loop over t chunks
        for t_chunk in range(grid_s):
            t_start = t_chunk * chunk_s
            t_end = jnp.minimum(t_start + chunk_s, S)
            
            # Load key and value chunks
            k_chunk = key_ref[b, h, t_start:t_end, :]
            v_chunk = value_ref[b, h, t_start:t_end, :]
            
            # Compute qk for this chunk
            # q_chunk: (s_size, D), k_chunk: (t_size, D)
            qk = jnp.dot(q_chunk.astype(jnp.float32), k_chunk.astype(jnp.float32).T)
            
            # Build position arrays for decay
            s_pos = jnp.arange(s_start, s_end, dtype=jnp.float32)
            t_pos = jnp.arange(t_start, t_end, dtype=jnp.float32)
            
            # Distance matrix: s_pos[:, None] - t_pos[None, :]
            dist = s_pos[:, None] - t_pos[None, :]
            
            # Causal mask
            causal = (dist >= 0).astype(jnp.float32)
            
            # Decay: gamma^(max(dist, 0)) * causal
            # gamma per head: 1 - exp2(-5 - h)
            gamma = 1.0 - jnp.exp2(-5.0 - jnp.float32(h))
            log_gamma = jnp.log(gamma)
            
            # max(dist, 0)
            max_dist = jnp.maximum(dist, 0.0)
            
            decay = jnp.exp(log_gamma * max_dist) * causal
            
            # Apply decay to qk
            qk = qk * decay
            
            # Accumulate output: qk @ v_chunk
            out_acc += jnp.dot(qk, v_chunk.astype(jnp.float32))
            
            # Accumulate sum of abs(qk) over last axis
            sum_acc += jnp.sum(jnp.abs(qk), axis=-1, keepdims=True)
        
        # Normalize
        sum_acc = jnp.maximum(sum_acc, 1.0)
        out_acc = out_acc / sum_acc
        
        # Write back
        out_ref[b, h, s_start:s_end, :] = out_acc.astype(query_ref.dtype)
    
    out_shape = jax.ShapeDtypeStruct(query.shape, query.dtype)
    
    return pl.pallas_call(
        retention_kernel,
        out_shape=out_shape,
        grid=(B, H, grid_s),
        in_specs=(
            pl.BlockSpec(lambda b, h, s: (b, h, s * chunk_s, 0), (B, H, chunk_s, D)),
            pl.BlockSpec(lambda b, h, s: (b, h, s * chunk_s, 0), (B, H, chunk_s, D)),
            pl.BlockSpec(lambda b, h, s: (b, h, s * chunk_s, 0), (B, H, chunk_s, D)),
        ),
        out_specs=pl.BlockSpec(lambda b, h, s: (b, h, s * chunk_s, 0), (B, H, chunk_s, D)),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel", "parallel", "parallel")),
    )(query, key, value)
