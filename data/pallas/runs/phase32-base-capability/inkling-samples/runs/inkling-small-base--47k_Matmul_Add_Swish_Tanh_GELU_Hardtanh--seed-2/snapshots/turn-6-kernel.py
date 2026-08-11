import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

def workload(x, weight, bias, add_value):
    block_m = 128
    block_n = 128

    def kernel(x_ref, w_ref, b_ref, add_ref, out_ref):
        # Accumulate matmul in float32
        x_f = x_ref[...].astype(jnp.float32)
        w_f = w_ref[...].astype(jnp.float32)
        acc = jnp.dot(x_f, w_f)

        # Add bias and add_value (broadcast over batch)
        b_f = b_ref[...].astype(jnp.float32)
        add_f = add_ref[...].astype(jnp.float32)
        acc = acc + b_f[jnp.newaxis, :]
        acc = acc + add_f[jnp.newaxis, :]

        # Swish
        acc = acc * jax.nn.sigmoid(acc)
        # Tanh
        acc = jnp.tanh(acc)
        # GELU
        acc = jax.nn.gelu(acc)
        # Hardtanh (clip to [-1, 1])
        acc = jnp.clip(acc, -1.0, 1.0)

        out_ref[...] = acc.astype(jnp.bfloat16)

    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        grid=(x.shape[0] // block_m, x.shape[1] // block_n),
        in_specs=(
            pl.BlockSpec((block_m, x.shape[1]), lambda i, j: (i * block_m, 0)),
            pl.BlockSpec((weight.shape[0], block_n), lambda i, j: (0, j * block_n)),
            pl.BlockSpec((block_n,), lambda i, j: (j * block_n,)),
            pl.BlockSpec((block_n,), lambda i, j: (j * block_n,)),
        ),
        out_specs=pl.BlockSpec((block_m, block_n), lambda i, j: (i * block_m, j * block_n)),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel", "parallel")),
    )(x, weight, bias, add_value)
