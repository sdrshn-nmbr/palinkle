import jax
import jax.numpy as jnp
import jax.pallas as pl
import jax.pallas as plp

def matmul_kernel(x_ref, weight_ref, bias_ref, out_ref):
    """Pallas kernel for Matmul + Scaling + ResidualAdd."""
    m = pl.program_id(0)
    n = pl.program_id(1)
    
    # Block shapes - multiples of 8 for bf16, 128 for vectorized dimensions
    block_m = 128
    block_n = 128
    block_k = 128
    
    # Accumulator in float32 for better precision
    acc = jnp.zeros((block_m, block_n), dtype=jnp.float32)
    
    # Tiled matmul over K dimension
    for k in range(0, 4096, block_k):
        # Load tiles
        x_tile = x_ref[m * block_m:(m + 1) * block_m, k:k + block_k]
        w_tile = weight_ref[k:k + block_k, n * block_n:(n + 1) * block_n]
        
        # Compute matmul and accumulate
        acc = acc + jnp.dot(x_tile.astype(jnp.float32), w_tile.astype(jnp.float32))
    
    # Add bias (broadcast along M dimension)
    bias_tile = bias_ref[n * block_n:(n + 1) * block_n]
    acc = acc + bias_tile
    
    # Save the unscaled result for residual add
    original = acc
    
    # Scale by 0.5
    acc = acc * 0.5
    
    # Add residual (original unscaled result)
    acc = acc + original
    
    # Store output as bfloat16
    out_ref[m * block_m:(m + 1) * block_m, n * block_n:(n + 1) * block_n] = acc.astype(jnp.bfloat16)


def workload(x, weight, bias):
    """Workload for Matmul + Scaling + ResidualAdd."""
    block_m = 128
    block_n = 128
    
    # Grid dimensions
    grid_m = x.shape[0] // block_m  # 16384 // 128 = 128
    grid_n = x.shape[1] // block_n  # 4096 // 128 = 32
    
    return pl.pallas_call(
        matmul_kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        grid=(grid_m, grid_n),
        in_specs=(
            pl.BlockSpec((block_m, 4096), lambda m, n: (m * block_m, 0)),
            pl.BlockSpec((4096, block_n), lambda m, n: (0, n * block_n)),
            pl.BlockSpec((4096,), lambda m, n: (0,)),
        ),
        out_specs=pl.BlockSpec((block_m, block_n), lambda m, n: (m * block_m, n * block_n)),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel")
        ),
    )(x, weight, bias)
