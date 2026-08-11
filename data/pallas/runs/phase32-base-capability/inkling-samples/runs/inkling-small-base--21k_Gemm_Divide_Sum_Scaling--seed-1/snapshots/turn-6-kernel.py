import jax
import jax.numpy as jnp
import jax.lax as lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

def workload(x, weight):
    def kernel(x_ref, w_ref, out_ref):
        # x_ref: (32, 8192), w_ref: (8192, 8192)
        # Compute dot: x_ref @ w_ref.T -> (32, 8192)
        # Use float32 accumulation
        x_f32 = x_ref[...].astype(jnp.float32)
        w_f32 = w_ref[...].astype(jnp.float32)
        # dot_general: contract x dim 1 with w.T dim 0 = w dim 1
        # So y = jnp.dot(x_ref, w_ref.T)
        y = jnp.dot(x_f32, w_f32.T)
        y = y / 2.0
        y = jnp.sum(y, axis=1, keepdims=True)
        y = y * 1.5
        out_ref[...] = y.astype(jnp.bfloat16)
    
    batch_size = x.shape[0]
    block = 32
    grid = (batch_size // block,)
    
    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((batch_size, 1), jnp.bfloat16),
        grid=grid,
        in_specs=(
            pl.BlockSpec((block, x.shape[1]), lambda i: (i, 0)),
            pl.no_block_spec,
        ),
        out_specs=pl.BlockSpec((block, 1), lambda i: (i, 0)),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel",)),
    )(x, weight)
