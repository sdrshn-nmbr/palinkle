import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl

def _kernel(x0_ref, x1_ref, out_ref):
    values = x0_ref[...].astype(jnp.float32)
    normalized = values * jax.lax.rsqrt(jnp.mean(jnp.square(values), axis=-1, keepdims=True) + 1e-5)
    out_ref[...] = normalized.astype(x0_ref.dtype) * x1_ref[...]

def workload(x0, x1):
    return pl.pallas_call(
        _kernel,
        out_shape=jax.ShapeDtypeStruct((4, 128, 512), jnp.float32),
        grid=(1,),
        in_specs=(
        pl.BlockSpec((4, 128, 512), lambda i: (0, 0, 0)),
        pl.BlockSpec((512,), lambda i: (0,))
        ),
        out_specs=pl.BlockSpec((4, 128, 512), lambda i: (0, 0, 0)),
    )(x0, x1)
