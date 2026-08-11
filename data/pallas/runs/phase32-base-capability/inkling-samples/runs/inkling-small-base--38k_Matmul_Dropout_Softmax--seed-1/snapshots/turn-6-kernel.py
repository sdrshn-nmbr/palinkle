import jax
import jax.numpy as jnp
import jax.lax as lax
import jax.experimental.pallas as pl
import jax.experimental.pallas.tpu as pltpu

def workload(x, weight, bias):
    def kernel(x_ref, weight_ref, bias_ref, out_ref):
        # Initialize scratch accumulation
        scratch = pltpu.VMEM((128, 8192), jnp.float32)
        # We need scratch_ref; but scratch_shapes passes refs
        pass
    
    # Actually let's design properly with scratch_shapes
    pass
