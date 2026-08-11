import jax
import jax.numpy as jnp
import jax.lax as lax
import jax.scipy.special as jss
import jax.experimental.pallas as pl
import jax.experimental.pallas.tpu as pltpu

def workload(x, conv_weight, conv_bias, gn_weight, gn_bias):
    def kernel(x_ref, w_ref, b_ref, gn_w_ref, gn_b_ref, out_ref):
        x = x_ref[...]
        w = w_ref[...]
        b = b_ref[...]
        gn_w = gn_w_ref[...]
        gn_b = gn_b_ref[...]

        groups = 16
        eps = 1e-5

        # Transpose x from NCHW to NHWC
        x_nhwc = jnp.transpose(x, (0, 2, 3, 1))
        # Transpose weight from OIHW to HWIO
        kernel_t = jnp.transpose(w, (2, 3, 1, 0))
        # Conv
        x_conv = lax.conv_general_dilated(
            x_nhwc, kernel_t,
            window_strides=(1, 1),
            padding="VALID",
            dimension_numbers=("NHWC", "HWIO", "NHWC")
        )
        # Add bias
        x_conv = x_conv + jnp.reshape(b, (1, 1, 1, -1))
        # Transpose back to NCHW
        x_conv = jnp.transpose(x_conv, (0, 3, 1, 2))

        N, C, H, W = x_conv.shape
        # GroupNorm reshape
        x_reshaped = jnp.reshape(x_conv, (N, groups, C // groups, H, W))
        mean = jnp.mean(x_reshaped, axis=(2, 3, 4), keepdims=True)
        var = jnp.var(x_reshaped, axis=(2, 3, 4), keepdims=True)
        x_reshaped = (x_reshaped - mean) / jnp.sqrt(var + eps)
        x = jnp.reshape(x_reshaped, (N, C, H, W))

        # Apply gn weight and bias
        x_norm = x * jnp.reshape(gn_w, (1, -1, 1, 1)) + jnp.reshape(gn_b, (1, -1, 1, 1))

        # Tanh
        x_tanh = jnp.tanh(x_norm)

        # HardSwish
        x_hard_swish = x_tanh * jnp.minimum(jnp.maximum(x_tanh + 3, 0), 6) / 6

        # Residual add
        x_res = x_conv + x_hard_swish

        # LogSumExp over channel axis (axis=1)
        x_logsumexp = jss.logsumexp(x_res, axis=1, keepdims=True)

        out_ref[...] = x_logsumexp

    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((128, 1, 126, 126), jnp.bfloat16),
        grid=(),
        in_specs=(
            pl.no_block_spec,
            pl.no_block_spec,
            pl.no_block_spec,
            pl.no_block_spec,
            pl.no_block_spec,
        ),
        out_specs=pl.no_block_spec,
    )(x, conv_weight, conv_bias, gn_weight, gn_bias)
