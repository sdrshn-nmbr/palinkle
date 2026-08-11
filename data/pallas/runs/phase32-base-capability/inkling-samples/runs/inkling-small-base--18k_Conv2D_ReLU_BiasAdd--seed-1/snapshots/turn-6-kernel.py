import jax
import jax.numpy as jnp
import jax.experimental.pallas as pl
import jax.experimental.pallas.tpu as pltpu

def workload(x, weight, conv_bias, bias):
    # Transpose x from NCHW to NHWC
    x_nhwc = jnp.transpose(x, (0, 2, 3, 1))
    # Transpose weight from OIHW to HWIO
    w_hwio = jnp.transpose(weight, (2, 3, 1, 0))

    def kernel(x_ref, w_ref, cb_ref, bias_ref, out_ref):
        x = x_ref[...]
        w = w_ref[...]
        # Conv: NHWC input, HWIO weight -> NHWC output
        y = jax.lax.conv_general_dilated(
            x, w,
            window_strides=(1, 1),
            padding='VALID',
            dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
        )
        cb = cb_ref[...]
        y = y + cb.reshape(1, 1, 1, -1)
        y = jax.nn.relu(y)
        # Transpose to NCHW for final output
        y = jnp.transpose(y, (0, 3, 1, 2))
        b = bias_ref[...]
        y = y + b
        out_ref[...] = y

    out_shape = jax.ShapeDtypeStruct((128, 128, 126, 126), jnp.bfloat16)
    return pl.pallas_call(
        kernel,
        out_shape=out_shape,
        grid=(16, 16, 2, 2),
        in_specs=(
            pl.BlockSpec((8, 65, 65, 64), lambda b, oc, h, w: (b * 8, h * 63, w * 63, 0)),
            pl.BlockSpec((3, 3, 64, 8), lambda b, oc, h, w: (0, 0, 0, oc * 8)),
            pl.BlockSpec((8,), lambda b, oc, h, w: (oc * 8,)),
            pl.BlockSpec((8, 1, 1), lambda b, oc, h, w: (oc * 8, 0, 0)),
        ),
        out_specs=pl.BlockSpec((8, 8, 63, 63), lambda b, oc, h, w: (b * 8, oc * 8, h * 63, w * 63)),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel", "parallel", "parallel", "parallel")),
    )(x_nhwc, w_hwio, conv_bias, bias)
