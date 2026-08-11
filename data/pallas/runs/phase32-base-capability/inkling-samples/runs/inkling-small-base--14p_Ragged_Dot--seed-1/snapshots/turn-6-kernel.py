import jax
import jax.numpy as jnp
import jax.experimental.pallas as pl
import jax.experimental.pallas.tpu as pltpu

def workload(x, weights):
    def ragged_dot_kernel(x_ref, w_ref, o_ref):
        # Load full K for this M/N tile
        x_f32 = x_ref[...].astype(jnp.float32)
        w_f32 = w_ref[...].astype(jnp.float32)
        # x_f32: (1, 128, 4096), w_f32: (1, 4096, 128)
        acc = jnp.dot(x_f32[0], w_f32[0])
        o_ref[...] = acc.astype(jnp.bfloat16)

    # Grid over groups (8), M tiles (1024/128=8), N tiles (14336/128=112)
    grid = (8, 8, 112)
    return pl.pallas_call(
        ragged_dot_kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        grid=grid,
        in_specs=(
            pl.BlockSpec((1, 128, 4096), lambda g, m, n: (g, m * 128, 0)),
            pl.BlockSpec((1, 4096, 128), lambda g, m, n: (g, 0, n * 128)),
        ),
        out_specs=pl.BlockSpec((1, 128, 128), lambda g, m, n: (g, m * 128, n * 128)),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel", "parallel", "parallel")),
    )(x, weights)
