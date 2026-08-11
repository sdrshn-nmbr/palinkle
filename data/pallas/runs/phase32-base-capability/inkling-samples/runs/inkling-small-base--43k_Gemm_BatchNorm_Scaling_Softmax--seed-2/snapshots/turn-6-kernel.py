import jax
import jax.numpy as jnp
import jax.lax as lax
import jax.experimental.pallas as pl
import jax.experimental.pallas.tpu as pltpu

def workload(x, weight, bias, bn_scale, bn_bias, bn_mean, bn_var, scale):
    def kernel(x_ref, weight_ref, bias_ref, bn_scale_ref, bn_bias_ref, bn_mean_ref, bn_var_ref, scale_ref, out_ref):
        # Load full arrays (single grid element)
        x_val = x_ref[...]
        w_val = weight_ref[...]
        b_val = bias_ref[...]
        
        # Matmul + bias
        y = jnp.dot(x_val, w_val) + b_val
        
        # Batch norm
        bn_eps = 1e-05
        x_normalized = (y - bn_mean_ref[...]) / jnp.sqrt(bn_var_ref[...] + bn_eps)
        
        # Scale and shift
        y = bn_scale_ref[...] * x_normalized + bn_bias_ref[...]
        
        # Scalar scale
        y = scale_ref[...] * y
        
        # Softmax over axis=1
        y = jax.nn.softmax(y, axis=1)
        
        out_ref[...] = y
    
    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        grid=(),
        in_specs=(
            pl.no_block_spec,
            pl.no_block_spec,
            pl.no_block_spec,
            pl.no_block_spec,
            pl.no_block_spec,
            pl.no_block_spec,
            pl.no_block_spec,
            pl.no_block_spec,
        ),
        out_specs=pl.no_block_spec,
    )(x, weight, bias, bn_scale, bn_bias, bn_mean, bn_var, scale)
