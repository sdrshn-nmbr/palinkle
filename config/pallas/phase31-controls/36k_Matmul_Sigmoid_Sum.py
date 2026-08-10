import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.pallas.ops.tpu.matmul import matmul


def _reduce_kernel(value_ref, bias_ref, out_ref):
    logits = value_ref[...].astype(jnp.float32) + bias_ref[...]
    out_ref[...] = jnp.sum(
        jax.nn.sigmoid(logits), axis=1, keepdims=True
    ).astype(jnp.bfloat16)


def workload(x, weight, bias):
    block_rows = 8
    values = matmul(
        x.astype(jnp.bfloat16),
        weight.astype(jnp.bfloat16),
        block_shape=(1024, 1024),
        block_k=1024,
        out_dtype=jnp.bfloat16,
    )
    return pl.pallas_call(
        _reduce_kernel,
        out_shape=jax.ShapeDtypeStruct((x.shape[0], 1), jnp.bfloat16),
        grid=(x.shape[0] // block_rows,),
        in_specs=(
            pl.BlockSpec((block_rows, values.shape[1]), lambda row: (row, 0)),
            pl.BlockSpec((bias.shape[0],), lambda row: (0,)),
        ),
        out_specs=pl.BlockSpec((block_rows, 1), lambda row: (row, 0)),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel",)),
    )(values, bias)
