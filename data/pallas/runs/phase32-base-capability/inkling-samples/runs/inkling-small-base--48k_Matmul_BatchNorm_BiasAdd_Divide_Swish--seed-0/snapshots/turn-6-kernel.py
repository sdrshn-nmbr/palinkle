import jax
import jax.numpy as jnp
import jax.lax as lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

def workload(x, weight, linear_bias, bn_scale, bn_bias, bn_mean, bn_var, bias):
    block_m = 128
    block_n = 128
    
    def kernel(x_ref, w_ref, lb_ref, bs_ref, bb_ref, bm_ref, bv_ref, bias_ref, out_ref):
        # Load tiles
        x_tile = x_ref[...].astype(jnp.float32)  # (block_m, 8192)
        w_tile = w_ref[...].astype(jnp.float32)  # (8192, block_n)
        
        # Matmul
        out_f32 = jnp.dot(x_tile, w_tile)  # (block_m, block_n)
        
        # Add linear_bias (block_n,)
        lb_tile = lb_ref[...].astype(jnp.float32)
        out_f32 = out_f32 + lb_tile[None, :]
        
        # BatchNorm
        bn_eps = jnp.float32(1e-05)
        bn_mean_tile = bm_ref[...].astype(jnp.float32)
        bn_var_tile = bv_ref[...].astype(jnp.float32)
        bn_scale_tile = bs_ref[...].astype(jnp.float32)
        bn_bias_tile = bb_ref[...].astype(jnp.float32)
        
        x_normalized = (out_f32 - bn_mean_tile[None, :]) / jnp.sqrt(bn_var_tile[None, :] + bn_eps)
        out_f32 = bn_scale_tile[None, :] * x_normalized + bn_bias_tile[None, :]
        
        # Add bias (scalar)
        bias_tile = bias_ref[...].astype(jnp.float32)
        out_f32 = out_f32 + bias_tile
        
        # Divide
        divide_value = jnp.float32(1.0)
        out_f32 = out_f32 / divide_value
        
        # Swish: x * sigmoid(x)
        out_f32 = out_f32 * jax.nn.sigmoid(out_f32)
        
        out_ref[...] = out_f32.astype(jnp.bfloat16)
    
    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        grid=(x.shape[0] // block_m, x.shape[1] // block_n),
        in_specs=(
            pl.BlockSpec((block_m, x.shape[1]), lambda i, j: (i * block_m, 0)),
            pl.BlockSpec((x.shape[1], block_n), lambda i, j: (0, j * block_n)),
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
