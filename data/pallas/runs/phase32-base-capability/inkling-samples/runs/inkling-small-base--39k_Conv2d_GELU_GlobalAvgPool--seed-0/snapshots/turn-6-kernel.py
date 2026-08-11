import jax
import jax.numpy as jnp
import jax.lax as lax
from jax import random
import jax.experimental.pallas as pl
import jax.experimental.pallas.tpu as pltpu

def workload(x, weight, bias):
    # Transpose inputs according to canonical semantics
    x_t = jnp.transpose(x, (0, 2, 3, 1))  # NCHW -> NHWC
    w_t = jnp.transpose(weight, (2, 3, 1, 0))  # OIHW -> HWIO
    
    # Bias reshape to (1,1,1,64)
    b_reshaped = jnp.reshape(bias, (1, 1, 1, -1))
    
    def kernel(x_ref, w_ref, b_ref, out_ref):
        # Load full block (batch=1, H=256, W=256, C=8)
        x_local = x_ref[...]
        w_local = w_ref[...]
        b_local = b_ref[...]
        
        # Conv: NHWC x HWIO -> NHWC
        conv_out = lax.conv_general_dilated(
            x_local,
            w_local,
            window_strides=(1, 1),
            padding="VALID",
            dimension_numbers=("NHWC", "HWIO", "NHWC"),
        )
        
        # Add bias
        out_local = conv_out + b_local
        
        # GELU
        out_local = jax.nn.gelu(out_local)
        
        # Global average pool over spatial axes 1,2
        out_local = jnp.mean(out_local, axis=(1, 2))
        
        out_ref[...] = out_local
    
    # Grid over batch dimension
    batch_size = x.shape[0]
    
    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((batch_size, 64), jnp.bfloat16),
        grid=(batch_size,),
        in_specs=(
            pl.BlockSpec((1, 256, 256, 8), lambda b: (b, 0, 0, 0)),
            pl.BlockSpec((3, 3, 8, 64), lambda b: (0, 0, 0, 0)),
            pl.BlockSpec((1, 1, 1, 64), lambda b: (0, 0, 0, 0)),
        ),
        out_specs=pl.BlockSpec((1, 64), lambda b: (b, 0)),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel",)),
    )(x_t, w_t, b_reshaped)
