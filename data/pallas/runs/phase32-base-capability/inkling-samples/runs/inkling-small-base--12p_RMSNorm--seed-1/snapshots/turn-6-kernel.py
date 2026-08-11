import jax
import jax.numpy as jnp
import jax.lax as lax
import jax.experimental.pallas as pl
import jax.experimental.pallas.tpu as pltpu

def workload(x, scale):
    def rmsnorm_kernel(x_ref, scale_ref, out_ref):
        # Read input block as float32 for computation
        x_f32 = jnp.asarray(x_ref[...], jnp.float32)
        # Compute mean of squares along last axis (emb_dim), keepdims
        mean2 = jnp.mean(jnp.square(x_f32), axis=-1, keepdims=True)
        # RMSNorm: x * rsqrt(mean2 + epsilon)
        normed = x_f32 * lax.rsqrt(mean2 + 1e-5)
        # Cast back to input dtype
        normed = jnp.asarray(normed, x_ref.dtype)
        # Apply scale (broadcast over batch/seq dims)
        s = jnp.asarray(scale_ref[...], jnp.float32)
        out_ref[...] = normed * s

    batch, seq_len, emb_dim = x.shape
    # Tile over batch and seq_len; process full emb_dim per block
    seq_block = 128
    grid = (batch, seq_len // seq_block)
    return pl.pallas_call(
        rmsnorm_kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        grid=grid,
        in_specs=(
            pl.BlockSpec((1, seq_block, emb_dim), lambda b, s: (b, s * seq_block, 0)),
            pl.BlockSpec((emb_dim,), lambda b, s: (0,)),
        ),
        out_specs=pl.BlockSpec((1, seq_block, emb_dim), lambda b, s: (b, s * seq_block, 0)),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel", "parallel")),
    )(x, scale)
