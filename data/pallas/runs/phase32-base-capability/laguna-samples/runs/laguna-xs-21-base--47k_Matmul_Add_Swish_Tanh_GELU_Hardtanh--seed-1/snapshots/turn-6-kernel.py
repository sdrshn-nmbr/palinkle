import jax
import jax.numpy as jnp
import pallas as pl
import pytpu

def matmul_add_swish_tanh_gelu_hardtanh_kernel(
    x_ref,
    weight_ref,
    bias_ref,
    add_value_ref,
    out_ref,
):
    """Pallas kernel for fused matmul + add + activations."""
    m = x_ref.shape[0]
    k = x_ref.shape[1]
    n = weight_ref.shape[1]
    
    # Block sizes for TPU
    block_m = 128
    block_n = 128
    block_k = 8  # For reduction
    
    # Get program indices
    i = pl.program_id(0)  # batch dimension
    j = pl.program_id(1)  # output feature dimension
    
    # Initialize accumulator in float32
    acc = jnp.zeros((block_m, block_n), dtype=jnp.float32)
    
    # Perform matmul with accumulation
    for kk in range(0, k, block_k):
        # Load x block
        x_block = x_ref[i * block_m : (i + 1) * block_m, kk : kk + block_k]
        # Load weight block
        w_block = weight_ref[kk : kk + block_k, j * block_n : (j + 1) * block_n]
        
        # Compute partial matmul
        acc = acc + jnp.dot(x_block, w_block)
    
    # Add bias and add_value
    result = acc + bias_ref[j * block_n : (j + 1) * block_n]
    result = result + add_value_ref[j * block_n : (j + 1) * block_n]
    
    # Apply swish: x * sigmoid(x)
    result = result * jax.nn.sigmoid(result)
    
    # Apply tanh
    result = jnp.tanh(result)
    
    # Apply gelu: x * Phi(x) + 0.5 * Phi((x - x^3/6) / sqrt(2*pi))
    result = jax.nn.gelu(result)
    
    # Apply hardtanh: clip to [-1, 1]
    result = jnp.clip(result, -1.0, 1.0)
    
    # Write output
    out_ref[i * block_m : (i + 1) * block_m, j * block_n : (j + 1) * block_n] = result.astype(x_ref.dtype)


def workload(x, weight, bias, add_value):
    """Fused kernel: matmul + add + swish + tanh + gelu + hardtanh."""
    # Output shape
    out_shape = jax.ShapeDtypeStruct((x.shape[0], weight.shape[1]), x.dtype)
    
    # Grid dimensions
    grid_m = x.shape[0] // 128
    grid_n = weight.shape[1] // 128
    
    return pl.pallas_call(
        matmul_add_swish_tanh_gelu_hardtanh_kernel,
        out_shape=out_shape,
        grid=(grid_m, grid_n),
        in_specs=(
            pl.BlockSpec((128, 8192), lambda idx: (idx[0] * 128, 0)),  # x
            pl.BlockSpec((8192, 128), lambda idx: (0, idx[1] * 128)),  # weight
            pl.BlockSpec((8192,), lambda idx: (idx[1] * 128,)),  # bias
            pl.BlockSpec((8192,), lambda idx: (idx[1] * 128,)),  # add_value
        ),
        out_specs=pl.BlockSpec((128, 128), lambda idx: (idx[0] * 128, idx[1] * 128)),
        compiler_params=pytpu.CompilerParams(
            dimension_semantics=("parallel", "parallel")
        ),
    )(x, weight, bias, add_value)
