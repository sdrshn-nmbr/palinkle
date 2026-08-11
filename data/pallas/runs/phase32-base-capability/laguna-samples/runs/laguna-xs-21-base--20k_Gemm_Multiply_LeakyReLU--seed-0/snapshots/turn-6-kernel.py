import jax
import jax.numpy as jnp
import jax.pallas as pl
import jax.pallas as plp
from jax.experimental import shard_map

def workload(x, weight, bias):
    """Gemm + Multiply + LeakyReLU kernel."""
    
    def kernel(matmul_out_ref, mult_out_ref, leaky_out_ref):
        # Matmul: x @ weight + bias
        # x is [4096, 8192], weight is [8192, 8192], bias is [8192]
        # Output of matmul is [4096, 8192]
        
        # Use jax.lax.dot for matmul with bias
        # We need to compute: x @ weight + bias
        # Since Pallas kernels work on blocks, we'll do the full computation
        
        # For simplicity, compute the entire operation in one go
        # matmul_out = x @ weight
        # Then add bias
        
        # Actually, let's do this step by step
        # Step 1: matmul
        result = jnp.dot(matmul_out_ref[...], weight)
        # Add bias
        result = result + bias
        matmul_out_ref[...] = result
        
        # Step 2: multiply by 2.0
        mult_out_ref[...] = matmul_out_ref[...] * 2.0
        
        # Step 3: LeakyReLU
        leaky_out_ref[...] = jnp.where(mult_out_ref[...] >= 0, mult_out_ref[...], mult_out_ref[...] * 0.1)
    
    # Define block shapes
    # x: [4096, 8192], weight: [8192, 8192], bias: [8192]
    # output: [4096, 8192]
    
    # Use a simple grid spec that processes the entire output
    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((4096, 8192), jnp.bfloat16),
        in_specs=(
            pl.BlockSpec((4096, 8192), lambda _: (0,)),  # x
            pl.BlockSpec((8192, 8192), lambda _: (0,)),  # weight
            pl.BlockSpec((8192,), lambda _: (0,)),       # bias
        ),
        out_specs=(
            pl.BlockSpec((4096, 8192), lambda _: (0,)),  # matmul output
            pl.BlockSpec((4096, 8192), lambda _: (0,)),  # mult output
            pl.BlockSpec((4096, 8192), lambda _: (0,)),  # leaky output
        ),
        compiler_params=plp.TPUCompilerParams(
            dimension_semantics=("parallel", "parallel")
        ),
    )(x, weight, bias)
