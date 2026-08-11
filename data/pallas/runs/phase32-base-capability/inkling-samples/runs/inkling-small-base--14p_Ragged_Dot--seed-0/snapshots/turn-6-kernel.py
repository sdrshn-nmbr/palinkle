import jax
import jax.numpy as jnp
import jax.lax as lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

def workload(x, weights):
    def ragged_dot_kernel(x_ref, w_ref, out_ref):
        # Load full M and K for x, full K and N tile for weights
        x_f32 = x_ref[...].astype(jnp.float32)
        w_f32 = w_ref[...].astype(jnp.float32)
        # x_f32 shape: (1, 1024, 4096), w_f32: (1, 4096, 128)
        # Compute matmul over last two dims
        acc = jnp.dot(x_f32[0], w_f32[0])  # (1024, 128)
        out_ref[...] = acc.astype(jnp.bfloat16).reshape(1, 1024, 128)

    return pl.pallas_call(
        ragged_dot_kernel,
        out_shape=jax.ShapeDtypeStruct((8, 1024, 14336), jnp.bfloat16),
        grid=(8, 14336 // 128),
        in_specs=(
            pl.BlockSpec((1, 1024, 4096), lambda g, n: (g, 0, 0)),
            pl.BlockSpec((1, 4096, 128), lambda g, n: (g, 0, n * 128)),
        ),
        out_specs=pl.BlockSpec((1, 1024, 128), lambda g, n: (g, 0, n * 128)),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel", "parallel")),
    )(x, weights)
