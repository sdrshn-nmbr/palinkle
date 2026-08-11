import jax
import jax.numpy as jnp
import jax.lax as lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

def workload(x, weight, conv_bias, in_weight, in_bias):
    def kernel(x_ref, w_ref, cb_ref, iw_ref, ib_ref, out_ref):
        x_val = x_ref[...]
        w_val = w_ref[...]
        cb_val = cb_ref[...]
        iw_val = iw_ref[...]
        ib_val = ib_ref[...]
        
        x_nhwc = jnp.transpose(x_val, (0, 2, 3, 1))
        kernel_t = jnp.transpose(w_val, (2, 3, 1, 0))
        x_conv = lax.conv_general_dilated(
            x_nhwc, kernel_t,
            window_strides=(1, 1),
            padding="VALID",
            dimension_numbers=("NHWC", "HWIO", "NHWC")
        )
        x_conv = x_conv + jnp.reshape(cb_val, (1, 1, 1, -1))
        x_nchw = jnp.transpose(x_conv, (0, 3, 1, 2))
        mean = jnp.mean(x_nchw, axis=(2, 3), keepdims=True)
        var = jnp.var(x_nchw, axis=(2, 3), keepdims=True)
        x_norm = (x_nchw - mean) / jnp.sqrt(var + 1e-5)
        x_norm = x_norm * jnp.reshape(iw_val, (1, -1, 1, 1)) + jnp.reshape(ib_val, (1, -1, 1, 1))
        out_ref[...] = x_norm / 2.0
    
    out_shape = jax.ShapeDtypeStruct((128, 128, 126, 126), jnp.bfloat16)
    return pl.pallas_call(
        kernel,
        out_shape=out_shape,
        grid=(1,),
        in_specs=(
            pl.BlockSpec(x.shape, lambda i: (0, 0, 0, 0)),
            pl.BlockSpec(weight.shape, lambda i: (0, 0, 0, 0)),
            pl.BlockSpec(conv_bias.shape, lambda i: (0,)),
            pl.BlockSpec(in_weight.shape, lambda i: (0,)),
            pl.BlockSpec(in_bias.shape, lambda i: (0,)),
        ),
        out_specs=pl.BlockSpec(out_shape.shape, lambda i: (0, 0, 0, 0)),
    )(x, weight, conv_bias, in_weight, in_bias)
