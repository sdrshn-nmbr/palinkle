import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl

BATCH, HEADS, SEQUENCE, HEAD_DIM = 2, 4, 128, 128

def _kernel(q_ref, k_ref, v_ref, out_ref):
    q = q_ref[0, 0].astype(jnp.float32)
    k = k_ref[0, 0].astype(jnp.float32)
    v = v_ref[0, 0].astype(jnp.float32)
    scores = jnp.dot(q, jnp.swapaxes(k, 0, 1)) / jnp.sqrt(jnp.asarray(HEAD_DIM, jnp.float32))
    positions = jnp.arange(SEQUENCE)
    scores = jnp.where(positions[:, None] >= positions[None, :], scores, -jnp.inf)
    maximum = jnp.max(scores, axis=-1, keepdims=True)
    numerator = jnp.exp(scores - maximum)
    probabilities = numerator / jnp.sum(numerator, axis=-1, keepdims=True)
    out_ref[0, 0] = jnp.dot(probabilities, v)

def workload(q, k, v):
    spec = pl.BlockSpec((1, 1, 128, 128), lambda b, h: (b, h, 0, 0))
    return pl.pallas_call(
        _kernel,
        out_shape=jax.ShapeDtypeStruct((BATCH, HEADS, SEQUENCE, HEAD_DIM), jnp.float32),
        grid=(BATCH, HEADS),
        in_specs=(spec, spec, spec),
        out_specs=spec,
    )(q, k, v)
