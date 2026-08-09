import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl

def _kernel(x0_ref, x1_ref, x2_ref, out_ref):
    values = jnp.dot(x0_ref[...], x1_ref[...], preferred_element_type=jnp.float32) + x2_ref[...]
    first = values * jnp.tanh(jnp.logaddexp(values, 0.0))
    out_ref[...] = first * jnp.tanh(jnp.logaddexp(first, 0.0))

def workload(x0, x1, x2):
    return pl.pallas_call(
        _kernel,
        out_shape=jax.ShapeDtypeStruct((384, 256), jnp.float32),
        grid=(1,),
        in_specs=(
        pl.BlockSpec((384, 256), lambda i: (0, 0)),
        pl.BlockSpec((256, 256), lambda i: (0, 0)),
        pl.BlockSpec((256,), lambda i: (0,))
        ),
        out_specs=pl.BlockSpec((384, 256), lambda i: (0, 0)),
    )(x0, x1, x2)
