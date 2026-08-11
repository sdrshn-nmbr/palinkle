import jax
import jax.numpy as jnp
import jax.lax as lax
import jax.experimental.pallas as pl
import jax.experimental.pallas.tpu as pltpu

def workload(x, weight, conv_bias, bias):
    def kernel(x_ref, w_ref, cb_ref, b_ref, out_ref):
        x_tile = x_ref[...]
        w_tile = w_ref[...]
        cb_tile = cb_ref[...]
        b_tile = b_ref[...]
        
        # x is NCHW: transpose to NHWC (0,2,3,1)
        x_nhwc = jnp.transpose(x_tile, (0, 2, 3, 1))
        # weight is OIHW: transpose to HWIO (2,3,1,0)
        w_hwio = jnp.transpose(w_tile, (2, 3, 1, 0))
        
        # Conv2D
        out_nhwc = lax.conv_general_dilated(
            x_nhwc,
            w_hwio,
            window_strides=(1, 1),
            padding="VALID",
            dimension_numbers=("NHWC", "HWIO", "NHWC"),
        )
        
        # Add conv_bias reshaped to (1,1,1,128)
        cb_reshaped = jnp.reshape(cb_tile, (1, 1, 1, -1))
        out_nhwc = out_nhwc + cb_reshaped
        
        # ReLU
        out_nhwc = jax.nn.relu(out_nhwc)
        
        # Transpose back to NCHW (0,3,1,2)
        out_nchw = jnp.transpose(out_nhwc, (0, 3, 1, 2))
        
        # Add bias (128,1,1)
        out_nchw = out_nchw + b_tile
        
        out_ref[...] = out_nchw
    
    batch_tile = 2
    grid = (128 // batch_tile,)
    
    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((128, 128, 126, 126), jnp.bfloat16),
        grid=grid,
        in_specs=(
            pl.BlockSpec((batch_tile, 64, 128, 128), lambda i: (i * batch_tile, 0, 0, 0)),
            pl.BlockSpec((128, 64, 3, 3), lambda i: (0, 0, 0, 0)),
            pl.no_block_spec,
            pl.no_block_spec,
        ),
        out_specs=pl.BlockSpec((batch_tile, 128, 126, 126), lambda i: (i * batch_tile, 0, 0, 0)),
    )(x, weight, conv_bias, bias)
