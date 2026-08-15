import jax
import jax.numpy as jnp
import jax.pallas as pl
import jax.pallas as plp
import jax.interpreters.pallas as pallas
from jax.interpreters import xla_extension as pltpu


def workload(x, weight, bias):
    """Matmul + Divide + GELU kernel."""
    block_size = 128
    
    def kernel(matmul_ref, bias_ref, out_ref):
        # Get program IDs for grid traversal
        m_block = pl.program_id(0)
        n_block = pl.program_id(1)
        
        # Accumulate in float32 for better precision
        acc = jnp.zeros((block_size,), dtype=jnp.float32)
        
        # Tile over the reduction dimension (k)
        for k_block in range(x.shape[1] // block_size):
            # Load matmul tile (block_size x block_size)
            a_tile = matmul_ref[m_block * block_size:(m_block + 1) * block_size,
                                k_block * block_size:(k_block + 1) * block_size]
            # Load weight tile (block_size x block_size)
            b_tile = matmul_ref[k_block * block_size:(k_block + 1) * block_size,
                                n_block * block_size:(n_block + 1) * block_size]
            
            # Matrix multiplication for this tile
            acc = acc + jnp.sum(a_tile * b_tile, axis=1).astype(jnp.float32)
        
        # Load bias and add
        bias_tile = bias_ref[n_block * block_size:(n_block + 1) * block_size]
        result = acc + bias_tile.astype(jnp.float32)
        
        # Divide by 10.0
        result = result / 10.0
        
        # Apply GELU
        result = jax.nn.gelu(result)
        
        # Write output
        out_ref[m_block * block_size:(m_block + 1) * block_size,
                n_block * block_size:(n_block + 1) * block_size] = result.astype(jnp.bfloat16)
    
    # Grid dimensions
    grid_m = x.shape[0] // block_size
    grid_n = x.shape[1] // block_size
    
    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((x.shape[0], x.shape[1]), jnp.bfloat16),
        grid=(grid_m, grid_n),
        in_specs=(
            pl.BlockSpec((block_size, block_size), lambda m, n, k: (m * block_size, k * block_size)),
            pl.BlockSpec((block_size, block_size), lambda m, n, k: (k * block_size, n * block_size)),
            pl.BlockSpec((block_size,), lambda m, n: (n * block_size,)),
        ),
        out_specs=pl.BlockSpec((block_size, block_size), lambda m, n: (m * block_size, n * block_size)),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel")
        ),
    )(x, weight, bias)
