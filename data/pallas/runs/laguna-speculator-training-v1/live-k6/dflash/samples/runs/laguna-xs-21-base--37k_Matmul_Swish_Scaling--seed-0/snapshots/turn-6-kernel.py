import jax
import jax.numpy as jnp
import jax.pallas as pl
import jax.pallas as plp
import jax.interpreters.tpu as pltpu


def matmul_swish_scaling_kernel(x_ref, weight_ref, bias_ref, out_ref, *, m, n, k):
    """Pallas kernel for Matmul + Swish + Scaling."""
    # Tile sizes - use multiples of 8 for bf16 and 128 for vectorization
    block_m = 128
    block_n = 128
    block_k = 8
    
    # Get program IDs
    i = pl.program_id(0)  # row block index
    j = pl.program_id(1)  # col block index
    
    # Accumulator in float32 for better precision
    acc = jnp.zeros((block_m, block_n), dtype=jnp.float32)
    
    # Matmul: accumulate over k dimension
    for kk in range(k // block_k):
        # Load blocks from x and weight
        x_block = x_ref[i * block_m:(i + 1) * block_m, kk * block_k:(kk + 1) * block_k]
        w_block = weight_ref[kk * block_k:(kk + 1) * block_k, j * block_n:(j + 1) * block_n]
        
        # Convert to float32 for accumulation
        x_block_f32 = x_block.astype(jnp.float32)
        w_block_f32 = w_block.astype(jnp.float32)
        
        # Matrix multiplication for this tile
        acc = acc + jnp.dot(x_block_f32, w_block_f32)
    
    # Add bias (broadcast along rows)
    bias_block = bias_ref[j * block_n:(j + 1) * block_n]
    acc = acc + bias_block.astype(jnp.float32)
    
    # Convert back to bfloat16
    acc_bf16 = acc.astype(jnp.bfloat16)
    
    # Swish activation: x * sigmoid(x)
    swish_out = acc_bf16 * jax.nn.sigmoid(acc_bf16)
    
    # Scaling by 2.0
    result = swish_out * 2.0
    
    # Write output
    out_ref[i * block_m:(i + 1) * block_m, j * block_n:(j + 1) * block_n] = result


def workload(x, weight, bias):
    """Matmul + Swish + Scaling workload."""
    m, k1 = x.shape  # 4096, 8192
    k2, n = weight.shape  # 8192, 8192
    
    # Grid dimensions - number of blocks in each dimension
    block_m = 128
    block_n = 128
    
    grid = (m // block_m, n // block_n)
    
    return pl.pallas_call(
        matmul_swish_scaling_kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        grid=grid,
        in_specs=(
            pl.BlockSpec((m, k1), lambda: (0, 0)),  # x
            pl.BlockSpec((k2, n), lambda: (0, 0)),  # weight
            pl.BlockSpec((n,), lambda: (0,)),      # bias
        ),
        out_specs=pl.BlockSpec((block_m, block_n), lambda i, j: (i * block_m, j * block_n)),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel")
        ),
        interpret=False,
    )(x, weight, bias)
