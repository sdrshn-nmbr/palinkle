import jax
import jax.numpy as jnp
import jax.pallas as pl
import jax.pallas as plp
import jax.interpreters.pallas as pj
import jax.numpy as jnp

def workload(x, weight, bias):
    """Matmul + Mish + Mish kernel."""
    block_size = 128  # Block size for tiling
    
    def kernel(x_ref, weight_ref, bias_ref, out_ref):
        # Get program IDs for grid indexing
        m = pl.program_id(0)  # Row index
        n = pl.program_id(1)  # Column index
        
        # Accumulator for matmul result in float32
        acc = 0.0
        
        # Tile along the reduction dimension (k)
        for k in range(x_ref.shape[1] // block_size):
            # Load blocks from x and weight
            x_block = x_ref[m, k * block_size:(k + 1) * block_size]
            w_block = weight_ref[k * block_size:(k + 1) * block_size, n]
            
            # Compute partial matmul in float32
            acc += jnp.sum(x_block.astype(jnp.float32) * w_block.astype(jnp.float32))
        
        # Add bias
        result = acc + bias_ref[n].astype(jnp.float32)
        
        # First Mish: x * tanh(softplus(x))
        result = result * jnp.tanh(jnp.softplus(result))
        
        # Second Mish: x * tanh(softplus(x))
        result = result * jnp.tanh(jnp.softplus(result))
        
        # Store result
        out_ref[m, n] = result.astype(jnp.bfloat16)
    
    # Grid dimensions
    grid_m = x.shape[0] // block_size
    grid_n = x.shape[1] // block_size
    
    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        grid=(grid_m, grid_n),
        in_specs=(
            pl.BlockSpec((grid_m, block_size), lambda m, n, k: (m, k * block_size)),
            pl.BlockSpec((block_size, grid_n), lambda m, n, k: (k * block_size, n)),
            pl.BlockSpec((block_size,), lambda m, n, k: (n,)),
        ),
        out_specs=pl.BlockSpec((block_size, block_size), lambda m, n, k: (m * block_size, n * block_size)),
        compiler_params=jax.pallas.tpu.CompilerParams(
            dimension_semantics=("parallel", "parallel")
        ),
    )(x, weight, bias)
