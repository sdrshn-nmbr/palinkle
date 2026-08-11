import jax
import jax.numpy as jnp
import jax.lax as lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

def workload(x, weight, bias):
    block_b = 128
    batch_size = x.shape[0]
    grid = (batch_size // block_b,)

    def kernel(x_ref, w_ref, b_ref, out_ref):
        # Load inputs as float32 for accumulation
        x_block = x_ref[...].astype(jnp.float32)
        w_block = w_ref[...].astype(jnp.float32)
        b_block = b_ref[...].astype(jnp.float32)
        # Matmul + bias
        res = jnp.dot(x_block, w_block) + b_block
        original = res
        scaled = res * 0.5
        out_ref[...] = (scaled + original).astype(jnp.bfloat16)

    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        grid=grid,
        in_specs=(
            pl.BlockSpec((block_b, x.shape[1]), lambda i: (i * block_b, 0)),
            pl.BlockSpec(weight.shape, lambda i: (0, 0)),
            pl.BlockSpec(bias.shape, lambda i: (0,)),
        ),
        out_specs=pl.BlockSpec((block_b, x.shape[1]), lambda i: (i * block_b, 0)),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel",)),
    )(x, weight, bias)
