import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl

SHAPE = (768, 384)

def _kernel(x0_ref, x1_ref, o_ref):
    o_ref[...] = x0_ref[...] + x1_ref[...]

def workload(x0, x1):
    spec = pl.BlockSpec((128, 128), lambda i, j: (i, j))
    return pl.pallas_call(
        _kernel,
        out_shape=jax.ShapeDtypeStruct(SHAPE, jnp.float32),
        grid=(SHAPE[0] // 128, SHAPE[1] // 128),
        in_specs=(spec, spec,),
        out_specs=spec,
    )(x0, x1)
