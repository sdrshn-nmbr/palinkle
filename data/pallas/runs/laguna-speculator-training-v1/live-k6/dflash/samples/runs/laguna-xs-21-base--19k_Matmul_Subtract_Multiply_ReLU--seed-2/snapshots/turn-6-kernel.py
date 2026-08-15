import jax
import jax.numpy as jnp
import jax.pallas as pl
import jax.pallas as plp
import jax.interpreters.tpu as pltpu

def workload(x, weight, bias):
    """Matmul + Subtract + Multiply + ReLU kernel."""
    
    block_size = 128
    
    def kernel(ref_x, ref_weight, ref_bias, ref_out):
        # Matmul: x @ weight
        # Accumulate in float32 for better precision
        matmul_result = jnp.dot(ref_x[...], ref_weight[...])
        
        # Add bias (broadcast along last axis)
        result = matmul_result + ref_bias[...]
        
        # Subtract 2.0
        result = result - 2.0
        
        # Multiply by 1.5
        result = result * 1.5
        
        # ReLU
        result = jax.nn.relu(result)
        
        ref_out[...] = result
    
    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        grid=(x.shape[0] // block_size, x.shape[1] // block_size),
        in_specs=(
            pl.BlockSpec((block_size, x.shape[1]), lambda i, j: (i * block_size, 0)),
            pl.BlockSpec((x.shape[1], block_size), lambda i, j: (0, j * block_size)),
            pl.BlockSpec((x.shape[1],), lambda i, j: (j * block_size,)),
        ),
        out_specs=pl.BlockSpec((block_size, block_size), lambda i, j: (i * block_size, j * block_size)),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel")
        ),
    )(x, weight, bias)
