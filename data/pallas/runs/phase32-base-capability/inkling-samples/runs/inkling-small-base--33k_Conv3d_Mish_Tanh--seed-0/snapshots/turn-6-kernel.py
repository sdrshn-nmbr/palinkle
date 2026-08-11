import jax
import jax.numpy as jnp
import jax.lax as lax
import jax.experimental.pallas as pl
import jax.experimental.pallas.tpu as pltpu

def workload(x, weight, bias):
    def kernel(x_ref, w_ref, b_ref, o_ref):
        x = jnp.transpose(x_ref[...], (0, 2, 3, 4, 1))
        w = jnp.transpose(w_ref[...], (2, 3, 4, 1, 0))
        x = lax.conv_general_dilated(
            x, w,
            window_strides=(1, 1, 1),
            padding=((0, 0), (0, 0), (0, 0)),
            dimension_numbers=("NDHWC", "DHWIO", "NDHWC")
        )
        x = x + jnp.reshape(b_ref[...], (1, 1, 1, 1, -1))
        x = x * jnp.tanh(jnp.log(1 + jnp.exp(x)))
        x = jnp.tanh(x)
        x = jnp.transpose(x, (0, 4, 1, 2, 3))
        o_ref[...] = x

    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((16, 64, 30, 62, 62), jnp.bfloat16),
        grid=(),
        in_specs=(
            pl.BlockSpec((16, 32, 32, 64, 64), lambda: (0, 0, 0, 0, 0)),
            pl.BlockSpec((64, 32, 3, 3, 3), lambda: (0, 0, 0, 0, 0)),
            pl.BlockSpec((64,), lambda: (0,)),
        ),
        out_specs=pl.BlockSpec((16, 64, 30, 62, 62), lambda: (0, 0, 0, 0, 0)),
    )(x, weight, bias)
