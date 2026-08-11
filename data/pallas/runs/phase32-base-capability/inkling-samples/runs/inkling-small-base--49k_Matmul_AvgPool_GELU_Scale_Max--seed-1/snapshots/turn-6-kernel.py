import jax
import jax.numpy as jnp
import jax.lax as lax
from jax import nn
import jax.experimental.pallas as pl
import jax.experimental.pallas.tpu as pltpu

def workload(x, weight, bias):
    def kernel(x_ref, w_ref, b_ref, out_ref):
        # Load inputs
        x_val = x_ref[...]
        w_val = w_ref[...]
        b_val = b_ref[...]
        
        # Matmul + bias
        y = jnp.dot(x_val, w_val) + b_val
        
        # Expand dims axis=1
        y = jnp.expand_dims(y, axis=1)
        
        # Avg pool: reduce_window with add, divide by 16
        y = lax.reduce_window(
            y,
            init_value=0.0,
            computation=lax.add,
            window_dimensions=(1, 1, 16),
            window_strides=(1, 1, 16),
            padding="VALID",
        ) / 16.0
        
        # Squeeze axis=1
        y = jnp.squeeze(y, axis=1)
        
        # GELU
        y = nn.gelu(y)
        
        # Scale
        y = y * 2.0
        
        # Max over axis=1
        y = jnp.max(y, axis=1)
        
        out_ref[...] = y
    
    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((4096,), jnp.bfloat16),
        grid=(),
        in_specs=(
            pl.no_block_spec,
            pl.no_block_spec,
            pl.no_block_spec,
        ),
        out_specs=pl.no_block_spec,
        interpret=False,
    )(x, weight, bias)
