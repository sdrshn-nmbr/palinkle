import jax
import jax.numpy as jnp
import jax.pallas as pl
import jax.pallas.lib as pllib
import jax.pallas.tpu as pltpu

def workload(x, weight, linear_bias, bn_scale, bn_bias, bn_mean, bn_var, bias):
    """Matmul + BatchNorm + BiasAdd + Divide + Swish kernel."""
    
    bn_eps = 1e-05
    divide_value = 1.0
    
    block_size = 128  # Block size for tiling
    
    def kernel(ref_x, ref_weight, ref_linear_bias, ref_bn_scale, ref_bn_bias, 
               ref_bn_mean, ref_bn_var, ref_bias, out_ref):
        # Get program IDs for grid traversal
        m = pl.program_id(0)  # batch dimension
        n = pl.program_id(1)  # output feature dimension
        
        # Accumulate matmul result in float32 for better precision
        acc = jnp.zeros((block_size,), dtype=jnp.float32)
        
        # Matmul: x[m, :] @ weight[:, n] + linear_bias[n]
        for k in range(x.shape[1] // block_size):
            x_block = ref_x[m * block_size:(m + 1) * block_size, k * block_size:(k + 1) * block_size]
            weight_block = ref_weight[k * block_size:(k + 1) * block_size, n * block_size:(n + 1) * block_size]
            acc = acc + jnp.dot(x_block.astype(jnp.float32), weight_block.astype(jnp.float32))
        
        # Add linear bias
        acc = acc + ref_linear_bias[n * block_size:(n + 1) * block_size].astype(jnp.float32)
        
        # Convert to bfloat16 for batch norm operations
        x_val = acc.astype(jnp.bfloat16)
        
        # BatchNorm: (x - bn_mean) / sqrt(bn_var + bn_eps) * bn_scale + bn_bias
        x_normalized = (x_val - ref_bn_mean[n * block_size:(n + 1) * block_size]) / \
                       jnp.sqrt(ref_bn_var[n * block_size:(n + 1) * block_size] + bn_eps)
        x_val = x_normalized * ref_bn_scale[n * block_size:(n + 1) * block_size] + \
                ref_bn_bias[n * block_size:(n + 1) * block_size]
        
        # BiasAdd: + bias
        x_val = x_val + ref_bias[0]
        
        # Divide: / divide_value
        x_val = x_val / divide_value
        
        # Swish: x * sigmoid(x)
        x_val = x_val * jax.nn.sigmoid(x_val)
        
        # Write output
        out_ref[m * block_size:(m + 1) * block_size, n * block_size:(n + 1) * block_size] = x_val
    
    # Grid dimensions based on input shapes
    grid = (x.shape[0] // block_size, x.shape[1] // block_size)
    
    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        grid=grid,
        in_specs=(
            pl.BlockSpec((block_size, block_size), lambda m, n: (m * block_size, 0)),  # x
            pl.BlockSpec((block_size, block_size), lambda m, n: (0, n * block_size)),  # weight
            pl.BlockSpec((block_size,), lambda m, n: (n * block_size,)),  # linear_bias
            pl.BlockSpec((block_size,), lambda m, n: (n * block_size,)),  # bn_scale
            pl.BlockSpec((block_size,), lambda m, n: (n * block_size,)),  # bn_bias
            pl.BlockSpec((block_size,), lambda m, n: (n * block_size,)),  # bn_mean
            pl.BlockSpec((block_size,), lambda m, n: (n * block_size,)),  # bn_var
            pl.BlockSpec((1,), lambda m, n: (0,)),  # bias
        ),
        out_specs=pl.BlockSpec((block_size, block_size), lambda m, n: (m * block_size, n * block_size)),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel")
        ),
    )(x, weight, linear_bias, bn_scale, bn_bias, bn_mean, bn_var, bias)
