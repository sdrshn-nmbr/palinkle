import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl

def _kernel(x0_ref, x1_ref, x2_ref, x3_ref, out_ref):
    gate = jax.nn.silu(jnp.dot(x0_ref[...], x1_ref[...]))
    up = jnp.dot(x0_ref[...], x2_ref[...])
    out_ref[...] = jnp.dot(gate * up, x3_ref[...])

def workload(x0, x1, x2, x3):
    return pl.pallas_call(
        _kernel,
        out_shape=jax.ShapeDtypeStruct((2, 128, 256), jnp.float32),
        grid=(1,),
        in_specs=(
        pl.BlockSpec((2, 128, 256), lambda i: (0, 0, 0)),
        pl.BlockSpec((256, 384), lambda i: (0, 0)),
        pl.BlockSpec((256, 384), lambda i: (0, 0)),
        pl.BlockSpec((384, 256), lambda i: (0, 0))
        ),
        out_specs=pl.BlockSpec((2, 128, 256), lambda i: (0, 0, 0)),
    )(x0, x1, x2, x3)
