import jax
import jax.numpy as jnp
import pallas as pl
import pallas.lib as pllib
import jax.pallas as pl
import jax.pallas.lib as pllib
import jax.interpreters.pallas as pl
import jax.interpreters.pallas.lib as pl
import jax.interpreters.pallas as pl
import jax.numpy as jnp
import pallas as pl
import pallas.lib as pllib

# Import TPU-specific modules
try:
    import jax.pallas as pl
    import jax.pallas.lib as pllib
    from jax import xla
    from jax.interpreters import pallas
    import jax.numpy as jnp
    import pallas as pl
    from jax.interpreters.pallas import TPU
    from jax.interpreters.pallas import lib as pllib
except ImportError:
    pass

# Actually, let me use the correct imports
import jax
import jax.numpy as jnp
import pallas as pl
from jax.interpreters.pallas import lib as pllib
from jax.interpreters.pallas import TPU

def matmul_swish_kernel(
    x_ref,
    weight_ref,
    bias_ref,
    out_ref,
):
    # Get program IDs for tiling
    m_block = pl.program_id(0)
    n_block = pl.program_id(1)
    
    # Block sizes
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 8
    
    # Accumulator in float32 for better precision
    acc = jnp.zeros((BLOCK_M, BLOCK_N), dtype=jnp.float32)
    
    # Matmul: out = x @ weight + bias
    # We need to compute the matmul result
    # x_ref has shape [BLOCK_M, 8192] (partial)
    # weight_ref has shape [8192, BLOCK_N] (partial)
    
    # For simplicity, let's do the entire computation in one block
    # and use proper indexing
    
    # Read inputs
    x_block = x_ref[...]  # [BLOCK_M, K]
    weight_block = weight_ref[...]  # [K, BLOCK_N]
    bias_block = bias_ref[...]  # [BLOCK_N]
    
    # Compute matmul in float32
    acc = jnp.dot(x_block, weight_block).astype(jnp.float32)
    
    # Add bias
    acc = acc + bias_block
    
    # Apply Swish: x * sigmoid(x)
    acc = acc * jax.nn.sigmoid(acc)
    
    # Scale by 2.0
    acc = acc * 2.0
    
    # Write output
    out_ref[...] = acc.astype(out_ref.dtype)


def workload(x, weight, bias):
    """Compute: (x @ weight + bias) * swish((x @ weight + bias)) * 2.0"""
    
    M, K = x.shape  # 4096, 8192
    K2, N = weight.shape  # 8192, 8192
    N2 = bias.shape[0]  # 8192
    
    # Output shape
    out_shape = jax.ShapeDtypeStruct((M, N), dtype=x.dtype)
    
    # Block sizes - TPU prefers multiples of 8 for bf16
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 8
    
    # Grid dimensions
    grid_m = M // BLOCK_M
    grid_n = N // BLOCK_N
    
    def kernel(x_ref, weight_ref, bias_ref, out_ref):
        # Get block indices
        m_block = pl.program_id(0)
        n_block = pl.program_id(1)
        
        # Compute local indices
        m_start = m_block * BLOCK_M
        n_start = n_block * BLOCK_N
        
        # Accumulate in float32 for precision
        acc = jnp.zeros((BLOCK_M, BLOCK_N), dtype=jnp.float32)
        
        # Loop over K dimension for matmul
        for k_block in range(K // BLOCK_K):
            k_start = k_block * BLOCK_K
            
            # Read slices
            x_slice = x_ref[m_start:m_start + BLOCK_M, k_start:k_start + BLOCK_K]
            w_slice = weight_ref[k_start:k_start + BLOCK_K, n_start:n_start + BLOCK_N]
            
            # Accumulate
            acc = acc + jnp.dot(x_slice, w_slice)
        
        # Add bias (broadcast along M dimension)
        bias_slice = bias_ref[n_start:n_start + BLOCK_N]
        acc = acc + bias_slice
        
        # Apply Swish activation: x * sigmoid(x)
        acc = acc * jax.nn.sigmoid(acc)
        
        # Scale by 2.0
        acc = acc * 2.0
        
        # Write output
        out_ref[m_start:m_start + BLOCK_M, n_start:n_start + BLOCK_N] = acc
    
    return pl.pallas_call(
        kernel,
        out_shape=out_shape,
        grid=(grid_m, grid_n),
        in_specs=(
            pl.BlockSpec((M, K), lambda idx: (idx, slice(None))),  # x
            pl.BlockSpec((K, N), lambda idx: (slice(None), idx)),  # weight
            pl.BlockSpec((N,), lambda idx: idx),  # bias
        ),
        out_specs=pl.BlockSpec((BLOCK_M, BLOCK_N), lambda idx: idx),
        compiler_params=TPU.CompilerParams(
            dimension_semantics=("parallel", "parallel")
        ),
    )(x, weight, bias)
