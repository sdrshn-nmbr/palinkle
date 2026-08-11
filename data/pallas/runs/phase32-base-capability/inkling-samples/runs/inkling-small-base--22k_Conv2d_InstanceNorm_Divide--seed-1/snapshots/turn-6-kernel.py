import jax
import jax.numpy as jnp
import jax.lax as lax
from jax.experimental import pallas as pl
import jax.experimental.pallas.tpu as pltpu

def workload(x, weight, conv_bias, in_weight, in_bias):
    # Tiling over batch (128) and out_channels (128)
    batch_tile = 8
    out_tile = 32
    
    def kernel(x_ref, w_ref, b_ref, iw_ref, ib_ref, out_ref):
        # Load tiles
        x_tile = x_ref[...]
        w_tile = w_ref[...]
        b_tile = b_ref[...]
        iw_tile = iw_ref[...]
        ib_tile = ib_ref[...]
        
        # x_tile: (B, 64, 128, 128) NCHW -> NHWC
        x_nhwc = jnp.transpose(x_tile, (0, 2, 3, 1))
        # w_tile: (O, 64, 3, 3) OIHW -> HWIO
        kernel_hwio = jnp.transpose(w_tile, (2, 3, 1, 0))
        
        # Conv
        x_conv = lax.conv_general_dilated(
            x_nhwc,
            kernel_hwio,
            window_strides=(1, 1),
            padding='VALID',
            dimension_numbers=('NHWC', 'HWIO', 'NHWC')
        )
        
        # Add conv bias (reshape to 1,1,1,O)
        x_conv = x_conv + jnp.reshape(b_tile, (1, 1, 1, -1))
        
        # Transpose back to NCHW for instance norm
        x_nchw = jnp.transpose(x_conv, (0, 3, 1, 2))
        
        # Instance norm over spatial dims (2,3)
        mean = jnp.mean(x_nchw, axis=(2, 3), keepdims=True)
        var = jnp.var(x_nchw, axis=(2, 3), keepdims=True)
        x_norm = (x_nchw - mean) / jnp.sqrt(var + 1e-5)
        
        # Scale and shift
        x_scaled = x_norm * jnp.reshape(iw_tile, (1, -1, 1, 1)) + jnp.reshape(ib_tile, (1, -1, 1, 1))
        
        # Divide by 2.0
        out_ref[...] = x_scaled / 2.0
    
    out_shape = jax.ShapeDtypeStruct((128, 128, 126, 126), x.dtype)
    
    return pl.pallas_call(
        kernel,
        out_shape=out_shape,
        grid=(128 // batch_tile, 128 // out_tile),
        in_specs=(
            pl.BlockSpec((batch_tile, 64, 128, 128), lambda b, o: (b, 0, 0, 0)),
            pl.BlockSpec((out_tile, 64, 3, 3), lambda b, o: (o, 0, 0, 0)),
            pl.BlockSpec((out_tile,), lambda b, o: (o,)),
            pl.BlockSpec((out_tile,), lambda b, o: (o,)),
            pl.BlockSpec((out_tile,), lambda b, o: (o,)),
        ),
        out_specs=pl.BlockSpec((batch_tile, out_tile, 126, 126), lambda b, o: (b, o, 0, 0)),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel", "parallel")),
    )(x, weight, conv_bias, in_weight, in_bias)
