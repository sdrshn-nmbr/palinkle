import jax
import jax.numpy as jnp
import jax.pallas as pl
import jax.pallas as plp
import jax.interpreters.pallas as plx
import jax.pallas.tpu as pltpu

def workload(x, weight, linear_bias, bn_scale, bn_bias, bn_mean, bn_var, bias):
    """Matmul + BatchNorm + BiasAdd + Divide + Swish fused kernel."""
    
    bn_eps = 1e-05
    divide_value = 1.0
    
    def kernel(ref_x, ref_weight, ref_linear_bias, ref_bn_scale, ref_bn_bias, 
               ref_bn_mean, ref_bn_var, ref_bias, out_ref):
        # Get program IDs for parallel dimensions
        m = pl.program_id(0)  # batch dimension
        n = pl.program_id(1)  # output feature dimension
        
        # Block sizes
        block_m = 128
        block_n = 128
        block_k = 8  # for matmul reduction
        
        # Matmul: x @ weight + linear_bias
        # Accumulate in float32 for better precision
        acc = jnp.zeros(block_n, dtype=jnp.float32)
        
        # Tile over K dimension
        for k in range(0, 8192, block_k):
            # Load x tile [block_m, block_k]
            x_tile = ref_x[m * block_m:(m + 1) * block_m, k:k + block_k].astype(jnp.float32)
            # Load weight tile [block_k, block_n]
            w_tile = ref_weight[k:k + block_k, n * block_n:(n + 1) * block_n].astype(jnp.float32)
            # Matmul and accumulate
            acc = acc + jnp.sum(x_tile * w_tile, axis=0)
        
        # Add linear bias
        acc = acc + ref_linear_bias[n * block_n:(n + 1) * block_n].astype(jnp.float32)
        
        # BatchNorm: (x - mean) / sqrt(var + eps) * scale + bias
        x_normalized = (acc - ref_bn_mean[n * block_n:(n + 1) * block_n].astype(jnp.float32)) / \
                       jnp.sqrt(ref_bn_var[n * block_n:(n + 1) * block_n].astype(jnp.float32) + bn_eps)
        x = x_normalized * ref_bn_scale[n * block_n:(n + 1) * block_n].astype(jnp.float32) + \
            ref_bn_bias[n * block_n:(n + 1) * block_n].astype(jnp.float32)
        
        # BiasAdd
        x = x + ref_bias[0].astype(jnp.float32)
        
        # Divide
        x = x / divide_value
        
        # Swish: x * sigmoid(x)
        x = x * jax.nn.sigmoid(x)
        
        # Write output
        out_ref[m * block_m:(m + 1) * block_m, n * block_n:(n + 1) * block_n] = x.astype(jnp.bfloat16)
    
    # Grid dimensions
    grid = (4096 // 128, 8192 // 128)  # (32, 64)
    
    # Block specs
    block_m = 128
    block_n = 128
    
    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((4096, 8192), jnp.bfloat16),
        grid=grid,
        in_specs=(
            pl.BlockSpec((block_m, 8), lambda m, n: (m * block_m, 0)),  # x
            pl.BlockSpec((8, block_n), lambda m, n: (0, n * block_n)),  # weight
            pl.BlockSpec((block_n,), lambda m, n: (n * block_n,)),  # linear_bias
            pl.BlockSpec((block_n,), lambda m, n: (n * block_n,)),  # bn_scale
            pl.BlockSpec((block_n,), lambda m, n: (n * block_n,)),  # bn_bias
            pl.BlockSpec((block_n,), lambda m, n: (n * block_n,)),  # bn_mean
            pl.BlockSpec((block_n,), lambda m, n: (n * block_n,)),  # bn_var
            pl.BlockSpec((1,), lambda m, n: (0,)),  # bias
        ),
        out_specs=pl.BlockSpec((block_m, block_n), lambda m, n: (m * block_m, n * block_n)),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel")
        ),
    )(x, weight, linear_bias, bn_scale, bn_bias, bn_mean, bn_var, bias)
