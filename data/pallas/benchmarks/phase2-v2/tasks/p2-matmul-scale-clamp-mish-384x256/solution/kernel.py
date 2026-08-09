import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl

def _kernel(x0_ref, x1_ref, x2_ref, out_ref):
    values = jnp.dot(x0_ref[...], x1_ref[...].T, preferred_element_type=jnp.float32) + x2_ref[...]
    values = jnp.clip(values * 4.0, -10.0, 10.0)
    reduced = jax.scipy.special.logsumexp(values, axis=1, keepdims=True)
    mish = reduced * jnp.tanh(jnp.logaddexp(reduced, 0.0))
    out_ref[...] = reduced * mish

def workload(x0, x1, x2):
    return pl.pallas_call(
        _kernel,
        out_shape=jax.ShapeDtypeStruct((384, 1), jnp.float32),
        grid=(1,),
        in_specs=(
        pl.BlockSpec((384, 256), lambda i: (0, 0)),
        pl.BlockSpec((256, 256), lambda i: (0, 0)),
        pl.BlockSpec((256,), lambda i: (0,))
        ),
        out_specs=pl.BlockSpec((384, 1), lambda i: (0, 0)),
    )(x0, x1, x2)
