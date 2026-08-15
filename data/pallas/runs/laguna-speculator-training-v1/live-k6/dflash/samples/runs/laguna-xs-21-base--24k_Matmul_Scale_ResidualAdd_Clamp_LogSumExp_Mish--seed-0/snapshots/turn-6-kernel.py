import jax
import jax.numpy as jnp
import jax.pallas as pl
import jax.pallas as pltpu


def workload(x, weight, bias):
    """TPU Pallas kernel for Matmul + Scale + ResidualAdd + Clamp + LogSumExp + Mish."""
    
    # Block size for tiling - use multiples of 8 for bf16 on TPU
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 128
    
    def kernel(x_ref, weight_ref, bias_ref, out_ref):
        # Get program IDs for tiling
        m_idx = pl.program_id(0)
        n_idx = pl.program_id(1)
        
        # Accumulator for matmul result in float32
        acc = jnp.zeros((BLOCK_M, BLOCK_N), dtype=jnp.float32)
        
        # Tiled matmul with bias
        for k_idx in range(x.shape[1] // BLOCK_K):
            x_block = x_ref[
                m_idx * BLOCK_M:(m_idx + 1) * BLOCK_M,
                k_idx * BLOCK_K:(k_idx + 1) * BLOCK_K
            ].astype(jnp.float32)
            
            weight_block = weight_ref[
                k_idx * BLOCK_K:(k_idx + 1) * BLOCK_K,
                n_idx * BLOCK_N:(n_idx + 1) * BLOCK_N
            ].astype(jnp.float32)
            
            acc = acc + jnp.dot(x_block, weight_block)
        
        # Add bias (broadcast along last dimension)
        bias_block = bias_ref[n_idx * BLOCK_N:(n_idx + 1) * BLOCK_N].astype(jnp.float32)
        acc = acc + bias_block
        
        # Scale by 2.0
        acc = acc * 2.0
        
        # Residual add: x + x (multiply by 2 again)
        acc = acc + acc
        
        # Clamp to [-10.0, 10.0]
        acc = jnp.clip(acc, -10.0, 10.0)
        
        # LogSumExp along axis 1 (last dimension), keepdims=True
        # This reduces the N dimension
        acc = jnp.logsumexp(acc, axis=1, keepdims=True)
        
        # Softplus: logaddexp(x, 0.0)
        softplus_x = jnp.logaddexp(acc, 0.0)
        
        # Mish: x * tanh(softplus_x)
        mish_x = acc * jnp.tanh(softplus_x)
        
        # Final multiply
        result = acc * mish_x
        
        # Write output (cast back to bfloat16)
        out_ref[...] = result.astype(x.dtype)
    
    # Grid dimensions
    grid_m = x.shape[0] // BLOCK_M
    grid_n = x.shape[1] // BLOCK_N
    
    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((x.shape[0], 1), x.dtype),
        grid=(grid_m, grid_n),
        in_specs=(
            pl.BlockSpec((BLOCK_M, BLOCK_K), lambda m, n, k: (m * BLOCK_M, k * BLOCK_K)),
            pl.BlockSpec((BLOCK_K, BLOCK_N), lambda m, n, k: (k * BLOCK_K, n * BLOCK_N)),
            pl.BlockSpec((BLOCK_N,), lambda m, n, k: (n * BLOCK_N,)),
        ),
        out_specs=pl.BlockSpec((BLOCK_M, 1), lambda m, n, k: (m * BLOCK_M, 0)),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel")
        ),
    )(x, weight, bias)
