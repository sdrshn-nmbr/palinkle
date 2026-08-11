import jax
import jax.numpy as jnp
import jax.lax as lax
from jax import nn
import jax.experimental.pallas as pl
import jax.experimental.pallas.tpu as pltpu

def workload(x, weight, bias):
    def kernel(x_ref, w_ref, b_ref, out_ref):
        row = x_ref[0, :]
        w = w_ref[...]
        b = b_ref[...]
        mat = jnp.dot(row, w) + b
        mat = mat.reshape(1, -1)
        mat = jnp.expand_dims(mat, axis=1)
        pooled = lax.reduce_window(
            mat,
            init_value=0.0,
            computation=lax.add,
            window_dimensions=(1, 1, 16),
            window_strides=(1, 1, 16),
            padding="VALID"
        ) / 16.0
        pooled = jnp.squeeze(pooled, axis=1)
        pooled = nn.gelu(pooled)
        pooled = pooled * 2.0
        result = jnp.max(pooled, axis=1)
        out_ref[0, :] = result
    
    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((4096,), jnp.bfloat16),
        grid=(4096,),
        in_specs=(
            pl.BlockSpec((1, 8192), lambda i: (i, 0)),
            pl.BlockSpec((8192, 8192), lambda i: (0, 0)),
            pl.BlockSpec((8192,), lambda i: (0,)),
        ),
        out_specs=pl.BlockSpec((1,), lambda i: (i,)),
    )(x, weight, bias)
