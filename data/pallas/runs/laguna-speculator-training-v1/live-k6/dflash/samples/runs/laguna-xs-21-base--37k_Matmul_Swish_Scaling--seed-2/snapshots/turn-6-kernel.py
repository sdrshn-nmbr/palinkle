import jax
import jax.numpy as jnp
import jax.pallas as pl
import jax.pallas as plp
import jax.interpreters.tpu as pltpu


def workload(x, weight, bias):
    """Matmul + Swish + Scaling kernel."""
    block_size = 128  # Block size for tiling
    
    def kernel(x_ref, weight_ref, bias_ref, out_ref):
        # Get program IDs for grid indexing
        m_idx = pl.program_id(0)  # Row index
        n_idx = pl.program_id(1)  # Column index
        
        # Compute tile bounds
        m_start = m_idx * block_size
        n_start = n_idx * block_size
        
        # Accumulate in float32 for better precision
        acc = jnp.zeros((block_size, block_size), dtype=jnp.float32)
        
        # Tiled matmul over k dimension
        for k_tile in range(x.shape[1] // block_size):
            k_start = k_tile * block_size
            
            # Load tiles
            x_tile = x_ref[
                m_start:m_start + block_size,
                k_start:k_start + block_size
            ]
            w_tile = weight_ref[
                k_start:k_start + block_size,
                n_start:n_start + block_size
            ]
            
            # Matmul and accumulate in float32
            acc = acc + jnp.dot(x_tile.astype(jnp.float32), w_tile.astype(jnp.float32))
        
        # Add bias (broadcast along rows)
        bias_tile = bias_ref[n_start:n_start + block_size]
        acc = acc + bias_tile[None, :]
        
        # Convert to bfloat16
        acc_bf16 = acc.astype(jnp.bfloat16)
        
        # Swish activation: x * sigmoid(x)
        swish_out = acc_bf16 * jax.nn.sigmoid(acc_bf16)
        
        # Scale by 2.0
        result = swish_out * 2.0
        
        # Write output
        out_ref[...] = result
    
    # Grid dimensions
    m_tiles = x.shape[0] // block_size
    n_tiles = x.shape[1] // block_size
    
    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        grid=(m_tiles, n_tiles),
        in_specs=(
            pl.BlockSpec((block_size, block_size), lambda m, n: (m * block_size, 0)),
            pl.BlockSpec((block_size, block_size), lambda m, n: (0, n * block_size)),
            pl.BlockSpec((block_size,), lambda m, n: (n * block_size,)),
        ),
        out_specs=pl.BlockSpec((block_size, block_size), lambda m, n: (m * block_size, n * block_size)),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel")
        ),
    )(x, weight, bias)
