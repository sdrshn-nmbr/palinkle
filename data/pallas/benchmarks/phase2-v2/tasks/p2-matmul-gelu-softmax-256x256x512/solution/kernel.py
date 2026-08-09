import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl

def _kernel(x0_ref, x1_ref, x2_ref, out_ref):
    values = jnp.dot(x0_ref[...], x1_ref[...], preferred_element_type=jnp.float32) + x2_ref[...]
    out_ref[...] = jax.nn.softmax(jax.nn.gelu(values), axis=1)

def workload(x0, x1, x2):
    return pl.pallas_call(
        _kernel,
        out_shape=jax.ShapeDtypeStruct((256, 512), jnp.float32),
        grid=(1,),
        in_specs=(
        pl.BlockSpec((256, 256), lambda i: (0, 0)),
        pl.BlockSpec((256, 512), lambda i: (0, 0)),
        pl.BlockSpec((512,), lambda i: (0,))
        ),
        out_specs=pl.BlockSpec((256, 512), lambda i: (0, 0)),
    )(x0, x1, x2)
