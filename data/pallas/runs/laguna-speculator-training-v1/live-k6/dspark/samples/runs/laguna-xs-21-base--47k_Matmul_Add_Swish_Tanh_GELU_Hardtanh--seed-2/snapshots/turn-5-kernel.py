import jax
import jax.numpy as jnp
import jax.pallas as pl
import jax.pallas as plp
import jax.interpreters.pallas as pallas
from jax.interpreters import xla_extension as pltpu


def workload(x, weight, bias, add_value):
    """
    Implements: Matmul + Add + Swish + Tanh + GELU + Hardtanh
    """
    batch_size, in_features = x.shape
    out_features = weight.shape[1]
    
    # Block size for matmul tiling
    block_size = 128
    
    def kernel(ref_x, ref_weight, ref_bias, ref_add_value, out_ref):
        # Get program IDs for grid indexing
        m = pl.program_id(0)  # batch dimension
        n = pl.program_id(1)  # output feature dimension
        
        # Initialize accumulator in float32 for better precision
        acc = jnp.zeros((out_features // block_size,) * block_size, dtype=jnp.float32)
        
        # Matmul: x[m, :] @ weight[:, n]
        # We need to tile over the reduction dimension (in_features)
        for k in range(0, in_features, block_size):
            # Load x tile: shape (block_size,)
            x_tile = ref_x[m, k:k+block_size]
            # Load weight tile: shape (block_size, block_size)
            w_tile = ref_weight[k:k+block_size, n*n_block_size:(n+1)*block_size]
            # Accumulate
            acc = acc + jnp.dot(x_tile.astype(jnp.float32), w_tile.astype(jnp.float32))
        
        # Convert back to bfloat16
        result = acc.astype(jnp.bfloat16)
        
        # Add bias
        result = result + ref_bias[n*n_block_size:(n+1)*block_size]
        
        # Add add_value
        result = result + ref_add_value[n*n_block_size:(n+1)*block_size]
        
        # Swish: x * sigmoid(x)
        result = result * jax.nn.sigmoid(result)
        
        # Tanh
        result = jnp.tanh(result)
        
        # GELU: approximate using x * sigmoid(1.702 * x)
        result = result * jax.nn.sigmoid(1.702 * result)
        
        # Hardtanh: clip to [-1, 1]
        result = jnp.clip(result, -1.0, 1.0)
        
        out_ref[m, n] = result
    
    # Grid dimensions
    grid_m = (batch_size + block_size - 1) // block_size
    grid_n = (out_features + block_size - 1) // block_size
    
    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((batch_size, out_features), jnp.bfloat16),
        grid=(grid_m, grid_n),
        in_specs=(
            pl.BlockSpec((batch_size, in_features), lambda: (0, 0)),
            pl.BlockSpec((in_features, out_features), lambda: (0, 0)),
            pl.BlockSpec((out_features,), lambda: 0),
            pl.BlockSpec((out_features,), lambda: 0),
        ),
        out_specs=pl.BlockSpec((batch_size, out_features), lambda: (0, 0)),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel")
        ),
    )(x, weight, bias, add_value)
