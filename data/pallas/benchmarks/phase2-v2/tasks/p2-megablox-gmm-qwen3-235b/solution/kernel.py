import jax
import jax.numpy as jnp
from jax.experimental.pallas.ops.tpu.megablox import gmm

def workload(lhs, rhs, group_sizes):
    return gmm(
        lhs,
        rhs,
        group_sizes,
        preferred_element_type=jnp.bfloat16,
        tiling=(256, 1024, 1024),
    )
