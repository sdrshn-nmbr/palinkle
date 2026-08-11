import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

BLOCK_M = 32
TILE_K = 1024
NUM_K_TILES = 8192 // TILE_K  # 8

def workload(x, weight, bias):
    # x: (4096, 8192), weight: (8192, 8192), bias: (8192,)
    # output: (4096, 1)
    def kernel(x_ref, w_ref, b_ref, out_ref):
        # Initialize scratch in float32
        # We use a scratch VMEM for accumulation
        # But scratch_shapes is passed to pallas_call; inside kernel we can use pl.program_id etc.
        # Actually to use scratch, we declare it in scratch_shapes and access via scratch_ref? 
        # Wait Pallas scratch is passed as additional refs? Let's check API.
        # From docs: scratch_shapes=() and inside kernel we can use pl.scratch? Not sure.
        # Actually in Pallas TPU, scratch is accessed via additional arguments? Let's inspect.
        pass
    
    # Actually let's look at how scratch is used in Pallas.
    pass
