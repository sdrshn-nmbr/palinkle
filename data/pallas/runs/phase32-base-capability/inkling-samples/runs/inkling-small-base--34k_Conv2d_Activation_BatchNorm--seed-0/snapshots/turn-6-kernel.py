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
        
        # Transpose x: (0,2,3,1) -> (N,H,W,C) to (N,C,H,W)? Wait instruction says transpose with (0,2,3,1)
        # Original x shape: (64,64,128,128) -> after transpose (0,2,3,1) -> (64,128,128,64)
        x = jnp.transpose(x, (0, 2, 3, 1))
        
        # Transpose weight: (2,3,1,0) -> (3,3,64,128)
        weight = jnp.transpose(conv_weight, (2, 3, 1, 0))
        
        # Conv: dimension_numbers = ("NHWC", "HWIO", "NHWC")
        # After transpose, x is (N,C,H,W) which is NCHW, but dimension_numbers says NHWC.
        # Wait, let's re-read: x is transposed with (0,2,3,1). Original x is (batch, in_channels, H, W) = (64,64,128,128).
        # After transpose (0,2,3,1): (batch, H, W, in_channels) = (64,128,128,64). Good, that's NHWC.
        # Weight is transposed with (2,3,1,0). Original conv_weight is (out_channels, in_channels, 3, 3) = (128,64,3,3).
        # After transpose (2,3,1,0): (3,3,64,128) = HWIO. Good.
        x = lax.conv_general_dilated(
            x, weight,
            window_strides=(1, 1),
            padding="VALID",
            dimension_numbers=("NHWC", "HWIO", "NHWC")
        )
        
        # Add bias reshaped to (1,1,1,128)
        x = x + jnp.reshape(conv_bias, (1, 1, 1, -1))
        
        # Transpose x with (0,3,1,2): (N,H,W,C) -> (N,C,H,W)
        x = jnp.transpose(x, (0, 3, 1, 2))
        
        # Softplus
        softplus_x = jnn.softplus(x)
        
        # Mish: tanh(softplus(x)) * x
        x = jnp.multiply(jnp.tanh(softplus_x), x)
        
        # BatchNorm mean over axes (0,2,3) keepdims=True
        mean = jnp.mean(x, axis=(0, 2, 3), keepdims=True)
        var = jnp.mean(jnp.power(x - mean, 2), axis=(0, 2, 3), keepdims=True)
        
        # Reshape bn_weight and bn_bias to (1,128,1,1)
        w = jnp.reshape(bn_weight, (1, -1, 1, 1))
        b = jnp.reshape(bn_bias, (1, -1, 1, 1))
        
        # BatchNorm
        x = (x - mean) / jnp.sqrt(var + eps)
        x = x * w + b
        
        # Output should be (64,128,126,126) which is NCHW. But instruction says output shape [64,128,126,126].
        # Wait, after batchnorm x is (N,C,H,W) = (64,128,126,126). That matches.
        out_ref[...] = x
    
    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((64, 128, 126, 126), jnp.bfloat16),
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
