import jax
import jax.numpy as jnp
import jax.pallas as pl
import jax.pallas as plp
import jax.interpreters.pallas as pj
from jax.interpreters import pallas as pl
import jax.numpy as jnp

def workload(x, weight, bias):
    """TPU Pallas kernel for Matmul + Scale + ResidualAdd + Clamp + LogSumExp + Mish"""
    
    # Block size for matmul - use multiples of 8 for bf16
    block_size = 128
    
    def kernel(ref_x, ref_weight, ref_bias, out_ref):
        # Get program IDs for grid layout
        m = pl.program_id(0)  # batch dimension
        k = pl.program_id(1)  # reduction dimension (for bias)
        
        # Matmul: x[m, :] @ weight[:, :] + bias
        # x shape: [4096, 8192], weight shape: [8192, 8192], bias shape: [8192]
        # Output shape: [4096, 8192]
        
        # Accumulate in float32 for better precision
        acc = jnp.zeros((8192,), dtype=jnp.float32)
        
        # Tile the matmul along the reduction dimension
        for i in range(0, 8192, block_size):
            x_block = ref_x[m, i:i+block_size].astype(jnp.float32)
            w_block = ref_weight[i:i+block_size, :].astype(jnp.float32)
            acc = acc + jnp.dot(x_block, w_block)
        
        # Add bias
        acc = acc + ref_bias.astype(jnp.float32)
        
        # Convert back to bfloat16
        x_val = acc.astype(jnp.bfloat16)
        
        # Scale by 2.0
        x_val = x_val * 2.0
        
        # ResidualAdd: x + x (multiply by 2 again effectively)
        x_val = x_val + x_val
        
        # Clamp between -10.0 and 10.0
        x_val = jnp.clip(x_val, -10.0, 10.0)
        
        # LogSumExp along axis 0 (since we're working with a single row)
        # keepdims=True means output stays as [8192, 1]
        # For a 1D array, logsumexp gives a scalar, but we need to keep dims
        x_val = jnp.expand_dims(jnp.logsumexp(x_val, axis=0, keepdims=True), axis=0)
        
        # Mish: x * tanh(logaddexp(x, 0.0))
        softplus_x = jnp.logaddexp(x_val, 0.0)
        mish_x = x_val * jnp.tanh(softplus_x)
        
        # Final output
        out_ref[...] = x_val * mish_x
    
    # Grid: one program per batch element
    grid = (x.shape[0],)
    
    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((x.shape[0], 1), x.dtype),
        grid=grid,
        in_specs=(
            pl.BlockSpec((1, x.shape[1]), lambda m,: (m, 0)),  # x: [batch, hidden]
            pl.BlockSpec((weight.shape[0], weight.shape[1]), lambda m,: (0, 0)),  # weight: full matrix
            pl.BlockSpec((bias.shape[0],), lambda m,: (0,)),  # bias: full vector
        ),
        out_specs=pl.BlockSpec((1, 1), lambda m,: (m, 0)),
        compiler_params=jax.pallas.tpu.CompilerParams(
            dimension_semantics=("parallel",)
        ),
    )(x, weight, bias)
