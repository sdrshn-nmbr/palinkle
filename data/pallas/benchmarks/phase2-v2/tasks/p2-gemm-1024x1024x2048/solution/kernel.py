import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

BM, BN, BK = 256, 256, 256

def _kernel(x_ref, y_ref, out_ref, acc_ref):
    @pl.when(pl.program_id(2) == 0)
    def _zero():
        acc_ref[...] = jnp.zeros_like(acc_ref)
    acc_ref[...] += jnp.dot(x_ref[...], y_ref[...], preferred_element_type=jnp.float32)
    out_ref[...] = acc_ref[...].astype(jnp.bfloat16)

def workload(x, y):
    return pl.pallas_call(
        _kernel,
        out_shape=jax.ShapeDtypeStruct((1024, 2048), jnp.bfloat16),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            in_specs=(
                pl.BlockSpec((BM, BK), lambda i, j, k: (i, k)),
                pl.BlockSpec((BK, BN), lambda i, j, k: (k, j)),
            ),
            out_specs=pl.BlockSpec((BM, BN), lambda i, j, k: (i, j)),
            grid=(1024 // BM, 2048 // BN, 1024 // BK),
            scratch_shapes=(pltpu.VMEM((BM, BN), jnp.float32),),
        ),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel", "arbitrary")
        ),
    )(x, y)
