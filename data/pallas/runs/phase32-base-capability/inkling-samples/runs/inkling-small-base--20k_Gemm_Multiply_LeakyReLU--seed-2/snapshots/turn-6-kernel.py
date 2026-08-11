import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

def workload(x, weight, bias):
    def kernel(x_ref, w_ref, b_ref, out_ref):
        x_block = x_ref[...]
        w_block = w_ref[...]
        b_block = b_ref[...]
        acc = jnp.dot(x_block.astype(jnp.float32), w_block.astype(jnp.float32))
        acc += b_block.astype(jnp.float32)
        acc *= 2.0
        acc = jnp.where(acc >= 0, acc, acc * 0.1)
        out_ref[...] = acc.astype(jnp.bfloat16)

    bm = 128
    bn = 128
    grid = (x.shape[0] // bm, x.shape[1] // bn)
    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        grid=grid,
        in_specs=(
            pl.BlockSpec((bm, x.shape[1]), lambda m, n: (m, 0)),
            pl.BlockSpec((weight.shape[0], bn), lambda m, n: (0, n)),
            pl.BlockSpec((bn,), lambda m, n: (n,)),
        ),
        out_specs=pl.BlockSpec((bm, bn), lambda m, n: (m, n)),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel", "parallel")),
    )(x, weight, bias)
