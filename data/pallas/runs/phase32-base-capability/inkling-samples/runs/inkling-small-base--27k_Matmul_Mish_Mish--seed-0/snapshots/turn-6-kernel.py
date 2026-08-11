import jax
import jax.numpy as jnp
import jax.lax as lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

def workload(x, weight, bias):
    def kernel(x_ref, w_ref, b_ref, out_ref):
        # Matmul in float32
        x_f = x_ref[...].astype(jnp.float32)
        w_f = w_ref[...].astype(jnp.float32)
        acc = jnp.dot(x_f, w_f)
        # Add bias
        b_f = b_ref[...].astype(jnp.float32)
        acc = acc + b_f
        # Mish 1: x * tanh(softplus(x))
        mish1 = acc * jnp.tanh(jax.nn.softplus(acc))
        # Mish 2
        mish2 = mish1 * jnp.tanh(jax.nn.softplus(mish1))
        out_ref[...] = mish2.astype(jnp.bfloat16)

    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        grid=(x.shape[0] // 128, x.shape[1] // 128),
        in_specs=(
            pl.BlockSpec((128, 8192), lambda i, j: (i * 128, 0)),
            pl.BlockSpec((8192, 128), lambda i, j: (0, j * 128)),
            pl.BlockSpec((128,), lambda i, j: (j * 128,)),
        ),
        out_specs=pl.BlockSpec((128, 128), lambda i, j: (i * 128, j * 128)),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel", "parallel")),
    )(x, weight, bias)
