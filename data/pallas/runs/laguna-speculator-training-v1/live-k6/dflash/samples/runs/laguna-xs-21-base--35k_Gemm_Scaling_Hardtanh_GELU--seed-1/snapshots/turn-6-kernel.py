import jax
import jax.numpy as jnp
import jax.pallas as pl
import jax.pallas as plp
import jax.interpreters.pallas as pallas
from jax.interpreters import pallas as pl
from jax.pallas import lib as pllib
import jax.numpy as jnp

def workload(x, weight, bias):
    """Gemm + Scaling + Hardtanh + GELU kernel."""
    batch_size, in_features = x.shape
    out_features = weight.shape[1]
    
    block_size = 128  # Block size for tiling
    
    def kernel(ref_x, ref_weight, ref_bias, ref_out):
        # Matmul: x @ weight + bias
        # Accumulate in float32 for better precision
        x_block = ref_x[...].astype(jnp.float32)
        weight_block = ref_weight[...].astype(jnp.float32)
        
        # Compute matmul
        result = jnp.dot(x_block, weight_block)
        
        # Add bias
        result = result + ref_bias[...].astype(jnp.float32)
        
        # Scale by 0.5
        result = result * 0.5
        
        # Hardtanh: clip to [-2, 2]
        result = jnp.clip(result, -2.0, 2.0)
        
        # GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        result = result * 0.5 * (1.0 + jnp.tanh(
            jnp.sqrt(2.0 / jnp.pi) * (result + 0.044715 * jnp.power(result, 3))
        ))
        
        # Convert back to bfloat16
        ref_out[...] = result.astype(x.dtype)
    
    # Grid specification for 2D output
    grid = (batch_size // block_size, out_features // block_size)
    
    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((batch_size, out_features), x.dtype),
        grid=grid,
        in_specs=(
            pl.BlockSpec((block_size, in_features), lambda i, j: (i * block_size, 0)),
            pl.BlockSpec((in_features, block_size), lambda i, j: (0, j * block_size)),
            pl.BlockSpec((block_size,), lambda i, j: (j * block_size,)),
        ),
        out_specs=pl.BlockSpec((block_size, block_size), lambda i, j: (i * block_size, j * block_size)),
        compiler_params=plp.CompilerParams(
            dimension_semantics=("parallel", "parallel")
        ),
    )(x, weight, bias)
