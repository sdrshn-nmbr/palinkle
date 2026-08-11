import jax
import jax.numpy as jnp
import jax.lax as lax
import jax.experimental.pallas as pl
import jax.experimental.pallas.tpu as pltpu

def workload(x, scale):
    def rmsnorm_kernel(x_ref, scale_ref, out_ref):
        x_f32 = jnp.asarray(x_ref[...], jnp.float32)
        mean2 = jnp.mean(lax.square(x_f32), axis=-1, keepdims=True)
        normed = x_f32 * lax.rsqrt(mean2 + 1e-5)
        normed = jnp.asarray(normed, x_ref.dtype)
        out_ref[...] = normed * scale_ref[...]

    return pl.pallas_call(
        rmsnorm_kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        grid=(x.shape[0], x.shape[1]),
        in_specs=(
            pl.BlockSpec((1, 1, x.shape[2]), lambda b, s: (b, s, 0)),
            pl.BlockSpec((x.shape[2],), lambda b, s: (0,)),
        ),
        out_specs=pl.BlockSpec((1, 1, x.shape[2]), lambda b, s: (b, s, 0)),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel", "parallel")),
    )(x, scale)
