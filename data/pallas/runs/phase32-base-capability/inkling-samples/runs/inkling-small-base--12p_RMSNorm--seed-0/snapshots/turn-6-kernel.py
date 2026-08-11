import jax
import jax.numpy as jnp
import jax.lax as lax
import jax.experimental.pallas as pl
import jax.experimental.pallas.tpu as pltpu

def workload(x, scale):
    def rmsnorm_kernel(x_ref, scale_ref, out_ref):
        x_f32 = jnp.asarray(x_ref[...], jnp.float32)
        mean2 = jnp.mean(lax.square(x_f32), axis=-1, keepdims=True)
        normed = x_f32 * lax.rsqrt(mean2 + 1e-05)
        normed = jnp.asarray(normed, x_ref.dtype)
        out_ref[...] = normed * jnp.asarray(scale_ref[...], x_ref.dtype)

    batch, seq_len, emb_dim = x.shape
    block_b = 1
    block_s = 128
    block_e = emb_dim

    return pl.pallas_call(
        rmsnorm_kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        grid=(batch // block_b, seq_len // block_s),
        in_specs=(
            pl.BlockSpec((block_b, block_s, block_e), lambda b, s: (b, s * block_s, 0)),
            pl.BlockSpec((block_e,), lambda b, s: (0,)),
        ),
        out_specs=pl.BlockSpec((block_b, block_s, block_e), lambda b, s: (b, s * block_s, 0)),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel", "parallel", "arbitrary")),
    )(x, scale)
