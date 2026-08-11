import jax
import jax.numpy as jnp
import pallas as pl
import triton.pallas as pltpu

def kernel(
    x_ref,
    weight_ref,
    linear_bias_ref,
    bn_scale_ref,
    bn_bias_ref,
    bn_mean_ref,
    bn_var_ref,
    bias_ref,
    out_ref,
):
    # Get program IDs for parallel execution
    m_idx = pl.program_id(0)  # batch dimension
    n_idx = pl.program_id(1)  # output feature dimension
    
    # Constants
    bn_eps = 1e-5
    divide_value = 1.0
    
    # Compute matmul result for this block
    # x_ref[m_idx, :] @ weight_ref[:, n_idx] + linear_bias_ref[n_idx]
    # We need to accumulate the matmul in float32 for numerical stability
    
    # Load x row and weight column for this block
    x_block = x_ref[m_idx, :]  # shape [8192]
    weight_block = weight_ref[:, n_idx]  # shape [8192]
    
    # Compute matmul
    matmul_result = jnp.dot(x_block, weight_block)  # scalar in float32
    
    # Add linear bias
    linear_bias_val = linear_bias_ref[n_idx]
    x_val = matmul_result + linear_bias_val
    
    # Batch normalization: (x - mean) / sqrt(var + eps)
    bn_mean_val = bn_mean_ref[n_idx]
    bn_var_val = bn_var_ref[n_idx]
    bn_scale_val = bn_scale_ref[n_idx]
    bn_bias_val = bn_bias_ref[n_idx]
    
    x_normalized = (x_val - bn_mean_val) / jnp.sqrt(bn_var_val + bn_eps)
    
    # Scale and shift
    x_val = bn_scale_val * x_normalized + bn_bias_val
    
    # Add bias (scalar broadcast)
    bias_val = bias_ref[0]  # shape [1]
    x_val = x_val + bias_val
    
    # Divide
    x_val = x_val / divide_value
    
    # Swish activation: x * sigmoid(x)
    x_val = x_val * jax.nn.sigmoid(x_val)
    
    # Write output
    out_ref[m_idx, n_idx] = x_val


def workload(
    x,
    weight,
    linear_bias,
    bn_scale,
    bn_bias,
    bn_mean,
    bn_var,
    bias,
):
    # Grid dimensions
    BLOCK_SIZE = 128
    
    # Output shape
    out_shape = jax.ShapeDtypeStruct((4096, 8192), jnp.bfloat16)
    
    return pl.pallas_call(
        kernel,
        out_shape,
        grid=(4096 // BLOCK_SIZE, 8192 // BLOCK_SIZE),
        in_specs=(
            pl.BlockSpec((BLOCK_SIZE, 8192), lambda m_idx, n_idx: (m_idx * BLOCK_SIZE, slice(None))),
            pl.BlockSpec((8192, BLOCK_SIZE), lambda m_idx, n_idx: (slice(None), n_idx * BLOCK_SIZE)),
            pl.BlockSpec((8192,), lambda m_idx, n_idx: (n_idx,)),
            pl.BlockSpec((8192,), lambda m_idx, n_idx: (n_idx,)),
            pl.BlockSpec((8192,), lambda m_idx, n_idx: (n_idx,)),
            pl.BlockSpec((8192,), lambda m_idx, n_idx: (n_idx,)),
            pl.BlockSpec((8192,), lambda m_idx, n_idx: (n_idx,)),
            pl.BlockSpec((1,), lambda m_idx, n_idx: (0,)),
        ),
        out_specs=pl.BlockSpec((BLOCK_SIZE, BLOCK_SIZE), lambda m_idx, n_idx: (m_idx * BLOCK_SIZE, n_idx * BLOCK_SIZE)),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel")
        ),
    )(x, weight, linear_bias, bn_scale, bn_bias, bn_mean, bn_var, bias)
