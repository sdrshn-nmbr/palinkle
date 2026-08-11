import jax
import jax.numpy as jnp
import jax.lax as lax
import jax.experimental.pallas as pl
import jax.experimental.pallas.tpu as pltpu

def workload(x, conv_weight, conv_bias, ln_weight, ln_bias):
    def kernel(x_ref, conv_weight_ref, conv_bias_ref, ln_weight_ref, ln_bias_ref, out_ref):
        # Load full arrays from refs
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

        # Transpose x: (0,2,3,4,1) -> (batch, d, h, w, channels)
        x = jnp.transpose(x, (0, 2, 3, 4, 1))

        # Transpose kernel: (2,3,4,1,0) -> (d,h,w,out,in)
        kernel = jnp.transpose(conv_weight, (2, 3, 4, 1, 0))
        # Flip axes 0,1,2
        kernel = jnp.flip(kernel, axis=(0, 1, 2))

        batch_size, d_in, h_in, w_in, channels = x.shape
        k = kernel_size
        d_dilated = d_in + (d_in - 1) * (stride - 1)
        h_dilated = h_in + (h_in - 1) * (stride - 1)
        w_dilated = w_in + (w_in - 1) * (stride - 1)

        x_dilated = jnp.zeros((batch_size, d_dilated, h_dilated, w_dilated, channels), dtype=x.dtype)
        # Set dilated positions
        # Use slicing assignment via jnp.pad or direct indexing
        # We can construct using jnp.zeros and then add at strides
        # Simpler: use jnp.pad and slice? Actually we need to place original at stride intervals.
        # We can do: x_dilated = x_dilated.at[:, ::stride, ::stride, ::stride, :].set(x)
        x_dilated = x_dilated.at[:, ::stride, ::stride, ::stride, :].set(x)
        x = x_dilated

        pad = (k - 1) - padding
        jax_padding = ((pad, pad), (pad, pad), (pad, pad))

        # conv_general_dilated: NDHWC input, DHWOI kernel, NDHWC output
        x = lax.conv_general_dilated(
            x,
            kernel,
            window_strides=(1, 1, 1),
            padding=jax_padding,
            dimension_numbers=("NDHWC", "DHWOI", "NDHWC"),
        )

        # Add bias reshaped to (1,1,1,1,64) or broadcast
        # conv_bias shape (64,)
        bias_reshaped = jnp.reshape(conv_bias, (1, 1, 1, 1, -1))
        x = x + bias_reshaped

        # Transpose to (0,4,1,2,3) -> (batch, channels, d, h, w)
        x = jnp.transpose(x, (0, 4, 1, 2, 3))

        # LayerNorm over last axis (channels)
        mean = jnp.mean(x, axis=-1, keepdims=True)
        var = jnp.mean((x - mean) ** 2, axis=-1, keepdims=True)
        x = (x - mean) / jnp.sqrt(var + eps)
        # ln_weight and ln_bias are (64,) -> reshape to (1,64,1,1,1)
        ln_w = jnp.reshape(ln_weight, (1, -1, 1, 1, 1))
        ln_b = jnp.reshape(ln_bias, (1, -1, 1, 1, 1))
        x = x * ln_w + ln_b

        # GELU
        x = jax.nn.gelu(x)

        # Scaling
        x = x * scaling_factor

        out_ref[...] = x

    out_shape = jax.ShapeDtypeStruct((32, 64, 32, 64, 64), jnp.dtype("bfloat16"))
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
