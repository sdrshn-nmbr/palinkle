import jax
import jax.numpy as jnp
import jax.pallas as pl
import jax.pallas as plp
import jax.interpreters.tpu as pltpu


def matmul_kernel(x_ref, weight_ref, bias_ref, out_ref):
    """Pallas kernel for Matmul + Scaling + ResidualAdd."""
    m = x_ref.shape[0]
    k = x_ref.shape[1]
    n = weight_ref.shape[1]
    
    # Accumulate in float32 for better precision
    acc = jnp.zeros((m, n), dtype=jnp.float32)
    
    # Perform matmul: x @ weight
    for i in range(k):
        acc = acc + x_ref[:, i] * weight_ref[i, :]
    
    # Add bias
    acc = acc + bias_ref[None, :]
    
    # Store original result for residual add
    original_x = acc
    
    # Scale by 0.5
    scaled = acc * 0.5
    
    # Residual add: scaled + original_x
    result = scaled + original_x
    
    # Write output (convert back to bfloat16)
    out_ref[...] = result.astype(x_ref.dtype)


def workload(x, weight, bias):
    """Matmul + Scaling + ResidualAdd workload."""
    block_size = 128  # Block size for tiling
    
    m, k = x.shape
    _, n = weight.shape
    
    # Grid dimensions
    grid_m = (m + block_size - 1) // block_size
    grid_n = (n + block_size - 1) // block_size
    
    def kernel(x_ref, weight_ref, bias_ref, out_ref):
        # Get block indices
        m_block = pl.program_id(0)
        n_block = pl.program_id(1)
        
        # Calculate block ranges
        m_start = m_block * block_size
        n_start = n_block * block_size
        m_end = min(m_start + block_size, m)
        n_end = min(n_start + block_size, n)
        
        # Extract blocks
        x_block = x_ref[:, :]  # Full x matrix
        weight_block = weight_ref[:, :]  # Full weight matrix
        bias_block = bias_ref[:]  # Full bias vector
        
        # Perform matmul with accumulation in float32
        acc = jnp.zeros((m_end - m_start, n_end - n_start), dtype=jnp.float32)
        
        # Matmul: x @ weight
        for i in range(k):
            x_col = x_block[m_start:m_end, i]
            w_row = weight_block[i, n_start:n_end]
            acc = acc + (x_col * w_row).astype(jnp.float32)
        
        # Add bias
        acc = acc + bias_block[n_start:n_end].astype(jnp.float32)
        
        # Store original for residual
        original = acc
        
        # Scale by 0.5
        scaled = acc * 0.5
        
        # Residual add
        result = scaled + original
        
        # Write output
        out_ref[...] = result.astype(x_ref.dtype)
    
    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        grid=(grid_m, grid_n),
        in_specs=(
            pl.BlockSpec((m, k), lambda _: (0, 0)),
            pl.BlockSpec((k, n), lambda _: (0, 0)),
            pl.BlockSpec((n,), lambda _: (0,)),
        ),
        out_specs=pl.BlockSpec((block_size, block_size), lambda i, j: (i * block_size, j * block_size)),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel")
        ),
    )(x, weight, bias)
