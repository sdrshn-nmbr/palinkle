import jax
import jax.numpy as jnp
import jax.pallas as pl
import jax.pallas as plp
import jax.interpreters.pallas as pj
import jaxlib.xla_extension as pltpu

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
    # Get program IDs for grid traversal
    m = pl.program_id(0)  # batch dimension
    n = pl.program_id(1)  # output feature dimension
    
    # Block size for reduction dimension
    k_block = 128
    
    # Accumulate matmul result in float32 for better precision
    acc = 0.0
    
    # Matmul: x[m, :] @ weight[:, n]
    for k in pl.range(x_ref.shape[1] // k_block):
        x_val = x_ref[m, k * k_block : (k + 1) * k_block]
        w_val = weight_ref[k * k_block : (k + 1) * k_block, n]
        acc += jnp.dot(x_val, w_val)
    
    # Handle remaining elements if any
    remaining = x_ref.shape[1] % k_block
    if remaining > 0:
        x_val = x_ref[m, -remaining:]
        w_val = weight_ref[-remaining:, n]
        acc += jnp.dot(x_val, w_val)
    
    # Add linear bias
    acc = acc + linear_bias_ref[n]
    
    # BatchNorm: (x - mean) / sqrt(var + eps) * scale + bias
    bn_eps = 1e-5
    x_normalized = (acc - bn_mean_ref[n]) / jnp.sqrt(bn_var_ref[n] + bn_eps)
    acc = bn_scale_ref[n] * x_normalized + bn_bias_ref[n]
    
    # Add bias
    acc = acc + bias_ref[0]
    
    # Divide by divide_value (1.0)
    divide_value = 1.0
    acc = acc / divide_value
    
    # Swish: x * sigmoid(x)
    acc = acc * jax.nn.sigmoid(acc)
    
    # Write output
    out_ref[m, n] = acc.astype(x_ref.dtype)


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
    batch_size = x.shape[0]
    out_features = weight.shape[1]
    
    # Use block size that's a multiple of 8 for bf16
    block_m = 128
    block_n = 128
    
    # Grid dimensions
    grid_m = (batch_size + block_m - 1) // block_m
    grid_n = (out_features + block_n - 1) // block_n
    
    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((batch_size, out_features), x.dtype),
        grid=(grid_m, grid_n),
        in_specs=(
            pl.BlockSpec((block_m, x.shape[1]), lambda m, n: (m * block_m, 0)),
            pl.BlockSpec((x.shape[1], block_n), lambda m, n: (0, n * block_n)),
            pl.BlockSpec((block_n,), lambda m, n: (n * block_n,)),
            pl.BlockSpec((block_n,), lambda m, n: (n * block_n,)),
            pl.BlockSpec((block_n,), lambda m, n: (n * block_n,)),
            pl.BlockSpec((block_n,), lambda m, n: (n * block_n,)),
            pl.BlockSpec((block_n,), lambda m, n: (n * block_n,)),
            pl.BlockSpec((1,), lambda m, n: (0,)),
        ),
        out_specs=pl.BlockSpec((block_m, block_n), lambda m, n: (m * block_m, n * block_n)),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel")
        ),
    )(x, weight, linear_bias, bn_scale, bn_bias, bn_mean, bn_var, bias)
