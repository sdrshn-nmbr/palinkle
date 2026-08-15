import jax
import jax.numpy as jnp
import pallas as pl
import jax.pallas as pltpu

def workload(x, weight, bias):
    """
    Matmul + Subtract + Multiply + ReLU kernel.
    
    Computes: relu((x @ weight + bias - 2.0) * 1.5)
    """
    batch_size = 4096
    in_features = 8192
    out_features = 8192
    
    # Block size for matmul tiling
    block = 128
    
    def kernel(ref_x, ref_weight, ref_bias, ref_out):
        # Get program indices
        m = pl.program_id(0)  # batch dimension
        n = pl.program_id(1)  # output feature dimension
        
        # Accumulator in float32 for better precision
        acc = jnp.zeros((block,), dtype=jnp.float32)
        
        # Matmul kernel: accumulate over k dimension
        for k in range(0, in_features, block):
            # Load blocks from x and weight
            x_block = ref_x[m, k:k+block]  # [block]
            w_block = ref_weight[k:k+block, n]  # [block]
            
            # Element-wise multiplication and accumulation
            acc = acc + x_block.astype(jnp.float32) * w_block.astype(jnp.float32)
        
        # Convert to bfloat16 and apply bias
        result = acc.astype(jnp.bfloat16) + ref_bias[n]
        
        # Subtract 2.0
        result = result - 2.0
        
        # Multiply by 1.5
        result = result * 1.5
        
        # ReLU
        result = jnp.maximum(result, 0.0)
        
        # Write output
        ref_out[m, n] = result
    
    # Grid dimensions
    grid = (batch_size // block, out_features // block)
    
    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((batch_size, out_features), jnp.bfloat16),
        grid=grid,
        in_specs=(
            pl.BlockSpec((batch_size, in_features), lambda m, n: (m, 0)),
            pl.BlockSpec((in_features, out_features), lambda m, n: (0, n)),
            pl.BlockSpec((out_features,), lambda m, n: (n,)),
        ),
        out_specs=pl.BlockSpec((batch_size, out_features), lambda m, n: (m, n)),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel")
        ),
    )(x, weight, bias)
