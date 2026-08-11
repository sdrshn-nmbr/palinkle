import jax
import jax.numpy as jnp
import pallas as pl
import pallas_core as plc
from pallas import TPU as pltpu

def kernel(
    x_ref,
    weight_ref,
    bias_ref,
    bn_scale_ref,
    bn_bias_ref,
    bn_mean_ref,
    bn_var_ref,
    scale_ref,
    out_ref,
):
    # Get program indices
    m = pl.program_id(0)
    n = pl.program_id(1)
    
    # Block size for matmul
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 8
    
    # Accumulator for matmul in float32
    acc = jnp.zeros((BLOCK_M, BLOCK_N), dtype=jnp.float32)
    
    # Matmul: x @ weight + bias
    # x is (batch_size, in_features), weight is (in_features, out_features)
    for k in range(8192 // BLOCK_K):
        x_block = x_ref[m * BLOCK_M:(m + 1) * BLOCK_M, k * BLOCK_K:(k + 1) * BLOCK_K]
        w_block = weight_ref[k * BLOCK_K:(k + 1) * BLOCK_K, n * BLOCK_N:(n + 1) * BLOCK_N]
        acc = acc + jnp.dot(x_block.astype(jnp.float32), w_block.astype(jnp.float32))
    
    # Add bias
    acc = acc + bias_ref[n * BLOCK_N:(n + 1) * BLOCK_N].reshape(1, -1)
    
    # BatchNorm: (x - mean) / sqrt(var + eps)
    bn_eps = 1e-5
    x_normalized = (acc - bn_mean_ref[n * BLOCK_N:(n + 1) * BLOCK_N].reshape(1, -1).astype(jnp.float32)) / \
                   jnp.sqrt(bn_var_ref[n * BLOCK_N:(n + 1) * BLOCK_N].reshape(1, -1).astype(jnp.float32) + bn_eps)
    
    # Scale: bn_scale * x_normalized + bn_bias
    scaled = bn_scale_ref[n * BLOCK_N:(n + 1) * BLOCK_N].reshape(1, -1).astype(jnp.float32) * x_normalized + \
             bn_bias_ref[n * BLOCK_N:(n + 1) * BLOCK_N].reshape(1, -1).astype(jnp.float32)
    
    # Final scale
    result = scale_ref[0] * scaled
    
    # Write output
    out_ref[m * BLOCK_M:(m + 1) * BLOCK_M, n * BLOCK_N:(n + 1) * BLOCK_N] = result.astype(x_ref.dtype)


def workload(x, weight, bias, bn_scale, bn_bias, bn_mean, bn_var, scale):
    batch_size = x.shape[0]
    in_features = x.shape[1]
    out_features = weight.shape[1]
    
    # Grid dimensions
    grid_m = batch_size // 128
    grid_n = out_features // 128
    
    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((batch_size, out_features), x.dtype),
        grid=(grid_m, grid_n),
        in_specs=(
            pl.BlockSpec((128, 8), lambda i: (i, 0)),  # x
            pl.BlockSpec((8, 128), lambda i: (0, i)),  # weight
            pl.BlockSpec((128,), lambda i: (0, i)),  # bias
            pl.BlockSpec((128,), lambda i: (0, i)),  # bn_scale
            pl.BlockSpec((128,), lambda i: (0, i)),  # bn_bias
            pl.BlockSpec((128,), lambda i: (0, i)),  # bn_mean
            pl.BlockSpec((128,), lambda i: (0, i)),  # bn_var
            pl.BlockSpec((1,), lambda i: (0,)),  # scale
        ),
        out_specs=pl.BlockSpec((128, 128), lambda i: (i, 0)),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel")
        ),
    )(x, weight, bias, bn_scale, bn_bias, bn_mean, bn_var, scale)
