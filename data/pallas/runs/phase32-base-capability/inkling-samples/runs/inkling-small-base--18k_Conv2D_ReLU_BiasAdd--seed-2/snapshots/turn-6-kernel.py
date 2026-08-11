import jax
import jax.numpy as jnp
import jax.lax as lax
from jax import nn
import jax.experimental.pallas as pl

def workload(x, weight, conv_bias, bias):
    def kernel(x_ref, weight_ref, conv_bias_ref, bias_ref, out_ref):
        x = x_ref[...]
        weight = weight_ref[...]
        conv_bias = conv_bias_ref[...]
        bias = bias_ref[...]
        
        x = jnp.transpose(x, (0, 2, 3, 1))
        kernel_w = jnp.transpose(weight, (2, 3, 1, 0))
        x = lax.conv_general_dilated(
            x, kernel_w,
            window_strides=(1, 1),
            padding="VALID",
            dimension_numbers=("NHWC", "HWIO", "NHWC")
        )
        x = x + jnp.reshape(conv_bias, (1, 1, 1, -1))
        x = nn.relu(x)
        x = jnp.transpose(x, (0, 3, 1, 2))
        x = x + bias
        out_ref[...] = x
    
    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((128, 128, 126, 126), jnp.bfloat16),
        grid=(),
        in_specs=(pl.no_block_spec, pl.no_block_spec, pl.no_block_spec, pl.no_block_spec),
        out_specs=pl.no_block_spec,
    )(x, weight, conv_bias, bias)
