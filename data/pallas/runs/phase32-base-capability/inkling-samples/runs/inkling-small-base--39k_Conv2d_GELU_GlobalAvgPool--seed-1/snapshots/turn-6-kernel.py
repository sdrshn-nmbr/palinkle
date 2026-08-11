import jax
import jax.numpy as jnp
import jax.lax as lax
from jax import nn
import jax.experimental.pallas as pl
import jax.experimental.pallas.tpu as pltpu

def workload(x, weight, bias):
    # Preserve original semantics: transpose inputs
    x_t = jnp.transpose(x, (0, 2, 3, 1))  # (128, 256, 256, 8)
    w_t = jnp.transpose(weight, (2, 3, 1, 0))  # (3, 3, 8, 64)
    
    batch_block = 8
    out_ch_block = 16
    
    def kernel(x_ref, w_ref, b_ref, out_ref):
        x_local = x_ref[...]
        w_local = w_ref[...]
        # Conv
        y = lax.conv_general_dilated(
            x_local,
            w_local,
            window_strides=(1, 1),
            padding="VALID",
            dimension_numbers=("NHWC", "HWIO", "NHWC"),
        )
        # Add bias
        b_local = b_ref[...].reshape(1, 1, 1, -1)
        y = y + b_local
        # GELU
        y = nn.gelu(y)
        # Global average pool over spatial axes 1,2
        y = jnp.mean(y, axis=(1, 2))
        out_ref[...] = y
    
    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((128, 64), jnp.bfloat16),
        grid=(128 // batch_block, 64 // out_ch_block),
        in_specs=(
            pl.BlockSpec((batch_block, 256, 256, 8), lambda i, j: (i * batch_block, 0, 0, 0)),
            pl.BlockSpec((3, 3, 8, out_ch_block), lambda i, j: (0, 0, 0, j * out_ch_block)),
            pl.BlockSpec((out_ch_block,), lambda i, j: (j * out_ch_block,)),
        ),
        out_specs=pl.BlockSpec((batch_block, out_ch_block), lambda i, j: (i * batch_block, j * out_ch_block)),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel", "parallel")),
    )(x_t, w_t, bias)
