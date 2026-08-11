import jax
import jax.numpy as jnp
import jax.lax as lax
import jax.nn as jnn
import jax.experimental.pallas as pl
import jax.experimental.pallas.tpu as pltpu

def workload(x, conv_weight, conv_bias, ln_weight, ln_bias):
    def kernel(x_ref, conv_weight_ref, conv_bias_ref, ln_weight_ref, ln_bias_ref, out_ref):
        x = x_ref[...]
        conv_weight = conv_weight_ref[...]
        conv_bias = conv_bias_ref[...]
        ln_weight = ln_weight_ref[...]
        ln_bias = ln_bias_ref[...]

        stride = 2
        padding = 1
        kernel_size = 4
        eps = 1e-05
        scaling_factor = 1.0

        x = jnp.transpose(x, (0, 2, 3, 4, 1))
        kernel = jnp.transpose(conv_weight, (2, 3, 4, 1, 0))
        kernel = jnp.flip(kernel, axis=(0, 1, 2))

        batch_size, d_in, h_in, w_in, channels = x.shape
        k = kernel_size
        d_dilated = d_in + (d_in - 1) * (stride - 1)
        h_dilated = h_in + (h_in - 1) * (stride - 1)
        w_dilated = w_in + (w_in - 1) * (stride - 1)

        x_dilated = jnp.zeros((batch_size, d_dilated, h_dilated, w_dilated, channels), dtype=x.dtype)
        x_dilated = x_dilated.at[:, ::stride, ::stride, ::stride, :].set(x)
        x = x_dilated

        pad = (k - 1) - padding
        jax_padding = ((pad, pad), (pad, pad), (pad, pad))

        x = lax.conv_general_dilated(
            x,
            kernel,
            window_strides=(1, 1, 1),
            padding=jax_padding,
            dimension_numbers=("NDHWC", "DHWOI", "NDHWC"),
        )

        x = x + jnp.reshape(conv_bias, (1, 1, 1, 1, -1))
        x = jnp.transpose(x, (0, 4, 1, 2, 3))

        mean = jnp.mean(x, axis=-1, keepdims=True)
        var = jnp.mean((x - mean) ** 2, axis=-1, keepdims=True)
        x = (x - mean) / jnp.sqrt(var + eps)
        x = x * ln_weight + ln_bias
        x = jnn.gelu(x)
        x = x * scaling_factor

        out_ref[...] = x

    out_shape = jax.ShapeDtypeStruct((32, 64, 32, 64, 64), x.dtype)
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
    )(x, conv_weight, conv_bias, ln_weight, ln_bias)
