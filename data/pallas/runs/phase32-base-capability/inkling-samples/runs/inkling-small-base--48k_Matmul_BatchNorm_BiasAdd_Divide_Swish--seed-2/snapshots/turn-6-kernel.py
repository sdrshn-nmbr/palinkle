import jax
import jax.numpy as jnp
import jax.lax as lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

def workload(x, weight, linear_bias, bn_scale, bn_bias, bn_mean, bn_var, bias):
    def kernel(x_ref, weight_ref, linear_bias_ref, bn_scale_ref, bn_bias_ref,
               bn_mean_ref, bn_var_ref, bias_ref, out_ref):
        # Matmul in float32
        x_f = x_ref.astype(jnp.float32)
        w_f = weight_ref.astype(jnp.float32)
        acc = jnp.dot(x_f, w_f)
        
        # Add linear bias (broadcast over batch dim)
        lb = linear_bias_ref.astype(jnp.float32)
        acc = acc + lb
        
        # BatchNorm
        mean = bn_mean_ref.astype(jnp.float32)
        var = bn_var_ref.astype(jnp.float32)
        scale = bn_scale_ref.astype(jnp.float32)
        b = bn_bias_ref.astype(jnp.float32)
        eps = 1e-05
        normalized = (acc - mean) / jnp.sqrt(var + eps)
        acc = scale * normalized + b
        
        # Add scalar bias
        bias_val = bias_ref.astype(jnp.float32)
        acc = acc + bias_val
        
        # Divide by 1.0
        acc = acc / 1.0
        
        # Swish: x * sigmoid(x)
        acc = acc * jax.nn.sigmoid(acc)
        
        out_ref[...] = acc.astype(jnp.bfloat16)
    
    batch_size = x.shape[0]
    out_features = x.shape[1]
    block_m = 128
    block_n = 128
    
    grid = (batch_size // block_m, out_features // block_n)
    
    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        grid=grid,
        in_specs=(
            pl.BlockSpec((block_m, out_features), lambda i, j: (i * block_m, 0)),
            pl.BlockSpec((out_features, block_n), lambda i, j: (0, j * block_n)),
            pl.BlockSpec((block_n,), lambda i, j: (j * block_n,)),
            pl.BlockSpec((block_n,), lambda i, j: (j * block_n,)),
            pl.BlockSpec((block_n,), lambda i, j: (j * block_n,)),
            pl.BlockSpec((block_n,), lambda i, j: (j * block_n,)),
            pl.BlockSpec((block_n,), lambda i, j: (j * block_n,)),
            pl.BlockSpec((1,), lambda i, j: (0,)),
        ),
        out_specs=pl.BlockSpec((block_m, block_n), lambda i, j: (i * block_m, j * block_n)),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel", "parallel")),
    )(x, weight, linear_bias, bn_scale, bn_bias, bn_mean, bn_var, bias)
