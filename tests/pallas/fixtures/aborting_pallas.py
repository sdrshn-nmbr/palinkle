import os

import jax
from jax.experimental import pallas as pl

SHAPE = (768, 384)


def _kernel(x0_ref, x1_ref, output_ref):
    output_ref[...] = x0_ref[...] + x1_ref[...]


def workload(x0, x1):
    spec = pl.BlockSpec((128, 128), lambda i, j: (i, j))
    return pl.pallas_call(
        _kernel,
        out_shape=jax.ShapeDtypeStruct(SHAPE, x0.dtype),
        grid=(SHAPE[0] // 128, SHAPE[1] // 128),
        in_specs=(spec, spec),
        out_specs=spec,
    )(x0, x1)


os.abort()
