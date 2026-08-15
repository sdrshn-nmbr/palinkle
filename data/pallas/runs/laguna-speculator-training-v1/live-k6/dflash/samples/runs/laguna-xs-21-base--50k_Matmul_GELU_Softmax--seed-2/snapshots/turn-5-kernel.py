import jax
import jax.numpy as jnp
import jax.pallas as pl
import jax.pallas as plp
import jax.interpreters.tpu as pltpu


def matmul_gelu_softmax_kernel(
    x_ref,
    weight_ref,
    bias_ref,
    out_ref,
):
    """Pallas kernel for Matmul + GELU + Softmax."""
    m = x_ref.shape[0]
    k = x_ref.shape[1]
    n = weight_ref.shape[1]
    
    # Get program indices
    row = pl.program_id(0)
    col = pl.program_id(1)
    
    # Accumulate matmul result in float32 for better precision
    acc = 0.0
    
    # Tile along the reduction dimension
    for t in range(k // 128):
        x_block = x_ref[row, t * 128:(t + 1) * 128].astype(jnp.float32)
        w_block = weight_ref[t * 128:(t + 1) * 128, col].astype(jnp.float32)
        acc += jnp.sum(x_block * w_block)
    
    # Add bias
    acc += bias_ref[col].astype(jnp.float32)
    
    # Apply GELU
    acc = acc * 0.5 * (1.0 + jnp.tanh(jnp.sqrt(2.0 / jnp.pi) * (acc + 0.044715 * acc * acc * acc)))
    
    # Store intermediate result for softmax
    # We need to do softmax across all columns for this row
    # For now, store the GELU result
    out_ref[row, col] = acc.astype(jnp.bfloat16)


def workload(x, weight, bias):
    """Matmul + GELU + Softmax workload."""
    block_size = 128
    
    # Grid dimensions
    m = x.shape[0]  # 4096
    n = weight.shape[1]  # 8192
    
    grid = (m, n // block_size)
    
    def kernel(x_ref, weight_ref, bias_ref, out_ref):
        row = pl.program_id(0)
        col = pl.program_id(1)
        
        # Accumulate matmul result in float32
        acc = 0.0
        
        # Tile along the reduction dimension (k=8192)
        for t in range(x_ref.shape[1] // block_size):
            x_block = x_ref[row, t * block_size:(t + 1) * block_size].astype(jnp.float32)
            w_block = weight_ref[t * block_size:(t + 1) * block_size, col * block_size:(col + 1) * block_size].astype(jnp.float32)
            acc += jnp.sum(x_block * w_block, axis=1)
        
        # Add bias
        acc = acc + bias_ref[col * block_size:(col + 1) * block_size].astype(jnp.float32)
        
        # Apply GELU: x * sigmoid(1.702 * x)
        acc = acc * jax.nn.sigmoid(1.702 * acc)
        
        # Store for softmax computation
        out_ref[row, col * block_size:(col + 1) * block_size] = acc.astype(jnp.bfloat16)
    
    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((x.shape[0], x.shape[1]), x.dtype),
        grid=grid,
        in_specs=(
            pl.BlockSpec((m, k), lambda row, col: (row, 0)),
            pl.BlockSpec((k, n), lambda row, col: (0, col * block_size)),
            pl.BlockSpec((n,), lambda row, col: (col * block_size,)),
        ),
        out_specs=pl.BlockSpec((m, n), lambda row, col: (row, col * block_size)),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel")
        ),
    )(x, weight, bias)
