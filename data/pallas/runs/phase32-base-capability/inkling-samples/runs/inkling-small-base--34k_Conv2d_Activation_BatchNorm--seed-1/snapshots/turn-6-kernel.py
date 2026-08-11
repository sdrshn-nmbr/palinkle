import jax
import jax.numpy as jnp
import jax.lax as lax
import jax.nn as jnn
import jax.experimental.pallas as pl
import jax.experimental.pallas.tpu as pltpu

def workload(x, conv_weight, conv_bias, bn_weight, bn_bias):
    def kernel(x_ref, conv_weight_ref, conv_bias_ref, bn_weight_ref, bn_bias_ref, out_ref):
        x = x_ref[...]
        conv_weight = conv_weight_ref[...]
        conv_bias = conv_bias_ref[...]
        bn_weight = bn_weight_ref[...]
        bn_bias = bn_bias_ref[...]
        
        eps = 1e-05
        
        # Transpose x: (0,2,3,1)
        x = jnp.transpose(x, (0, 2, 3, 1))
        
        # Transpose weight: (2,3,1,0)
        weight = jnp.transpose(conv_weight, (2, 3, 1, 0))
        
        # Conv
        x = lax.conv_general_dilated(
            x, weight,
            window_strides=(1, 1),
            padding="VALID",
            dimension_numbers=("NHWC", "HWIO", "NHWC")
        )
        
        # Add bias reshaped (1,1,1,-1)
        x = x + jnp.reshape(conv_bias, (1, 1, 1, -1))
        
        # Transpose x: (0,3,1,2)
        x = jnp.transpose(x, (0, 3, 1, 2))
        
        # Mish activation: tanh(softplus(x)) * x
        softplus_x = jnn.softplus(x)
        x = jnp.multiply(jnp.tanh(softplus_x), x)
        
        # BatchNorm mean over axes (0,2,3) keepdims=True
        mean = jnp.mean(x, axis=(0, 2, 3), keepdims=True)
        var = jnp.mean(jnp.square(x - mean), axis=(0, 2, 3), keepdims=True)
        
        # Reshape bn weights/biases (1,-1,1,1)
        w = jnp.reshape(bn_weight, (1, -1, 1, 1))
        b = jnp.reshape(bn_bias, (1, -1, 1, 1))
        
        x = (x - mean) / jnp.sqrt(var + eps)
        x = x * w
        x = x + b
        
        out_ref[...] = x
    
    out_shape = jax.ShapeDtypeStruct((64, 128, 126, 126), x.dtype)
    return pl.pallas_call(
        kernel,
        out_shape=out_shape,
        grid=(),
        in_specs=(
            pl.no_block_spec,
            pl.no_block_spec,
            pl.no_block_spec,
            pl.no_block_spec,
            pl.no_block_spec,
        ),
        out_specs=pl.no_block_spec,
    )(x, conv_weight, conv_bias, bn_weight, bn_bias)
