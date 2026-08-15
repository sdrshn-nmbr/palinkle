import jax
import jax.numpy as jnp
import jax.pallas as pl
import jax.pallas as plp
import jax.interpreters.tpu as pltpu


def matmul_dropout_softmax_kernel(
    x_ref,
    weight_ref,
    bias_ref,
    out_ref,
):
    """Pallas kernel for Matmul + Dropout (skipped) + Softmax."""
    # Get program IDs for grid layout
    m = pl.program_id(0)  # batch dimension
    n = pl.program_id(1)  # output feature dimension
    
    # Block sizes - use multiples of 8 for bf16 and 128 for vectorization
    block_m = 128
    block_n = 128
    
    # Compute matmul: x[m, :] @ weight.T -> result[m, n]
    # Accumulate in float32 for better precision
    result = jnp.zeros((block_n,), dtype=jnp.float32)
    
    # Tile over the reduction dimension (in_features = 8192)
    for k in range(8192 // block_m):
        x_block = x_ref[m, k * block_m:(k + 1) * block_m].astype(jnp.float32)
        weight_block = weight_ref[k * block_m:(k + 1) * block_m, n].astype(jnp.float32)
        result = result + jnp.dot(x_block, weight_block)
    
    # Add bias
    bias_block = bias_ref[n].astype(jnp.float32)
    result = result + bias_block
    
    # Cast back to bf16 for softmax computation
    result_bf16 = result.astype(jnp.bfloat16)
    
    # Softmax along axis 0 (the feature dimension within this block)
    # For the full softmax, we need to handle the entire row
    # But since we're tiling, we need to compute softmax across all blocks
    
    # For simplicity, compute softmax on the full row
    # We need to gather all blocks first
    pass


def workload(x, weight, bias):
    """Matmul + Dropout (skipped in inference) + Softmax."""
    batch_size = x.shape[0]  # 4096
    in_features = x.shape[1]  # 8192
    out_features = weight.shape[1]  # 8192
    
    # Use a simpler approach: compute the full matmul + bias, then softmax
    # Use grid spec for the output shape
    
    block_size = 128
    
    def kernel(x_ref, weight_ref, bias_ref, out_ref):
        m = pl.program_id(0)
        n_start = pl.program_id(1) * block_size
        n_end = min(n_start + block_size, out_features)
        
        # Compute matmul for this block
        # x[m, :] @ weight[:, n_start:n_end] + bias[n_start:n_end]
        x_row = x_ref[m, :]  # (8192,)
        weight_block = weight_ref[:, n_start:n_end]  # (8192, block_size)
        bias_block = bias_ref[n_start:n_end]  # (block_size,)
        
        # Compute matmul in float32 for precision
        result = jnp.dot(x_row.astype(jnp.float32), weight_block.astype(jnp.float32))
        result = result + bias_block.astype(jnp.float32)
        
        # Store intermediate result
        out_ref[m, n_start:n_end] = result.astype(jnp.bfloat16)
    
    # First compute matmul + bias
    grid = (batch_size, (out_features + block_size - 1) // block_size)
    
    result = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((batch_size, out_features), jnp.bfloat16),
        grid=grid,
        in_specs=(
            pl.BlockSpec((batch_size, in_features), lambda m, n: (m, 0)),
            pl.BlockSpec((in_features, out_features), lambda m, n: (0, n * block_size)),
            pl.BlockSpec((out_features,), lambda m, n: (n * block_size,)),
        ),
        out_specs=pl.BlockSpec((batch_size, out_features), lambda m, n: (m, n * block_size)),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel")
        ),
    )(x, weight, bias)
    
    # Apply softmax along axis 1
    return jax.nn.softmax(result, axis=1)
