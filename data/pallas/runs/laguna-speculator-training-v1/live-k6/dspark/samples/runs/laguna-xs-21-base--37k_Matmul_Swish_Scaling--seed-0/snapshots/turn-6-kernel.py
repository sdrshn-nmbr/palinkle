import jax
import jax.numpy as jnp
import jax.pallas as pl
import jax.pallas as plp
import jax.interpreters.pallas as pallas
from jax.interpreters import xla_bridge
from jax import xla

# TPU-specific imports
try:
    from jax.interpreters import pallas as pallas_mod
    pltpu = pallas_mod.tpu
except ImportError:
    pltpu = None

def workload(x, weight, bias):
    """Matmul + Swish + Scaling kernel for TPU."""
    
    # Block size for TPU - need multiples of 8 for bf16 and 128 for vectorized dims
    block_size = 128
    
    # Grid dimensions based on input shapes
    grid = (x.shape[0] // block_size, x.shape[1] // block_size)
    
    def matmul_swish_scaling_kernel(
        x_ref, weight_ref, bias_ref, out_ref
    ):
        # Get program IDs for tiling
        m_block = pl.program_id(0)
        n_block = pl.program_id(1)
        
        # Compute tile offsets
        m_start = m_block * block_size
        n_start = n_block * block_size
        
        # Initialize accumulator in float32 for better precision
        acc = jnp.zeros((block_size, block_size), dtype=jnp.float32)
        
        # Tiled matmul along K dimension
        for k_block in range(weight.shape[0] // block_size):
            k_start = k_block * block_size
            
            # Load tiles
            x_tile = x_ref[m_start:m_start + block_size, k_start:k_start + block_size]
            w_tile = weight_ref[k_start:k_start + block_size, n_start:n_start + block_size]
            
            # Accumulate in float32
            acc = acc + jnp.dot(x_tile.astype(jnp.float32), w_tile.astype(jnp.float32))
        
        # Add bias (broadcast along M dimension)
        bias_tile = bias_ref[n_start:n_start + block_size]
        acc = acc + bias_tile[None, :]
        
        # Convert to bfloat16 for swish computation
        acc_bf16 = acc.astype(jnp.bfloat16)
        
        # Swish activation: x * sigmoid(x)
        swish_out = acc_bf16 * jax.nn.sigmoid(acc_bf16)
        
        # Scaling by 2.0
        result = swish_out * 2.0
        
        # Write output
        out_ref[m_start:m_start + block_size, n_start:n_start + block_size] = result
    
    # Define block specs for inputs and outputs
    in_specs = (
        pl.BlockSpec((block_size, block_size), lambda m, n: (m * block_size, n * block_size)),
        pl.BlockSpec((block_size, block_size), lambda m, n: (n * block_size, m * block_size)),
        pl.BlockSpec((block_size,), lambda m, n: (n * block_size,)),
    )
    
    out_specs = pl.BlockSpec((block_size, block_size), lambda m, n: (m * block_size, n * block_size))
    
    return pl.pallas_call(
        matmul_swish_scaling_kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        grid=grid,
        in_specs=in_specs,
        out_specs=out_specs,
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel")
        ),
    )(x, weight, bias)
