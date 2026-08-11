import jax
import jax.numpy as jnp
import jax.lax as lax
import jax.experimental.pallas as pl
import jax.experimental.pallas.tpu as pltpu

def workload(x, weight, bias):
    def kernel(x_ref, weight_ref, bias_ref, out_ref):
        x = jnp.transpose(x_ref[...], (0, 2, 3, 4, 1))
        kernel = jnp.transpose(weight_ref[...], (2, 3, 4, 1, 0))
        x = lax.conv_general_dilated(
            x,
            kernel,
            window_strides=(1, 1, 1),
            padding=((0, 0), (0, 0), (0, 0)),
            dimension_numbers=("NDHWC", "DHWIO", "NDHWC"),
        )
        bias_reshaped = jnp.reshape(bias_ref[...], (1, 1, 1, 1, -1))
        x = x + bias_reshaped
        x = x * jnp.tanh(jnp.log(1 + jnp.exp(x)))
        x = jnp.tanh(x)
        x = jnp.transpose(x, (0, 4, 1, 2, 3))
        out_ref[...] = x

    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((16, 64, 30, 62, 62), jnp.bfloat16),
        grid=(),
        in_specs=(
            pl.no_block_spec,
            pl.no_block_spec,
            pl.no_block_spec,
        ),
        out_specs=pl.no_block_spec,
    )(x, weight, bias)
