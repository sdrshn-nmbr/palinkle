import jax
import jax.numpy as jnp
import jax.experimental.pallas as pl
import jax.experimental.pallas.tpu as pltpu

def workload(x, weight, bias):
    def kernel(x_ref, weight_ref, bias_ref, out_ref):
        b = pl.program_id(0)
        # Load x slice using pl.load with dynamic slice
        x_slice = pl.load(x_ref, (pl.dslice(b * 128, 128), pl.dslice(0, 8192)))
        w = pl.load(weight_ref, (pl.dslice(0, 8192), pl.dslice(0, 8192)))
        b_ref = pl.load(bias_ref, (pl.dslice(0, 8192),))
        res = jnp.dot(x_slice.astype(jnp.float32), w.astype(jnp.float32).T)
        res = res + b_ref.astype(jnp.float32)
        res = res * 2.0
        res = res + res
        res = jnp.clip(res, -10.0, 10.0)
        res = jax.scipy.special.logsumexp(res, axis=1, keepdims=True)
        softplus_x = jnp.logaddexp(res, 0.0)
        mish_x = res * jnp.tanh(softplus_x)
        res = res * mish_x
        pl.store(out_ref, (pl.dslice(b * 128, 128), pl.dslice(0, 1)), res.astype(jnp.bfloat16))

    batch_size = x.shape[0]
    block = 128
    grid = (batch_size // block,)
    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((batch_size, 1), jnp.bfloat16),
        grid=grid,
        in_specs=(
            pl.BlockSpec((block, x.shape[1]), lambda b: (b * block, 0)),
            pl.BlockSpec(weight.shape, lambda b: (0, 0)),
            pl.BlockSpec(bias.shape, lambda b: (0,)),
        ),
        out_specs=pl.BlockSpec((block, 1), lambda b: (b * block, 0)),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel",)),
    )(x, weight, bias)
