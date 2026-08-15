import jax
import jax.numpy as jnp
import jax.pallas as pl
import jax.pallas as plp
import jax.interpreters.tpu as pltpu

def workload(x, weight, bias):
    """Matmul + Softmax kernel for TPU."""
    batch_size, in_features = x.shape
    out_features = weight.shape[1]
    
    # Block size for TPU - multiples of 8 for bf16
    block_size = 128
    
    def matmul_softmax_kernel(x_ref, weight_ref, bias_ref, out_ref):
        # Get program IDs for grid layout
        m = pl.program_id(0)  # batch dimension
        n = pl.program_id(1)  # output feature dimension
        
        # Compute matmul result for this block
        # x[m, :] @ weight[:, n] -> result[m, n]
        # Accumulate in float32 for better precision
        acc = 0.0
        
        for k in range(in_features // block_size):
            x_block = x_ref[m, k * block_size:(k + 1) * block_size].astype(jnp.float32)
            w_block = weight_ref[k * block_size:(k + 1) * block_size, n].astype(jnp.float32)
            acc += jnp.sum(x_block * w_block)
        
        # Add bias
        acc += bias_ref[n].astype(jnp.float32)
        
        # Store intermediate result for softmax computation
        # We need to compute softmax across all n values for each m
        # This requires a second pass or shared memory
        
        # For simplicity, let's use a different approach:
        # First compute all matmul results, then softmax
        
        out_ref[m, n] = acc.astype(jnp.bfloat16)
    
    # Grid for matmul computation
    grid = (batch_size // block_size, out_features // block_size)
    
    # First compute matmul + bias
    def matmul_kernel(x_ref, weight_ref, bias_ref, out_ref):
        m = pl.program_id(0)
        n = pl.program_id(1)
        
        acc = 0.0
        for k in range(in_features // block_size):
            x_block = x_ref[m, k * block_size:(k + 1) * block_size].astype(jnp.float32)
            w_block = weight_ref[k * block_size:(k + 1) * block_size, n].astype(jnp.float32)
            acc += jnp.sum(x_block * w_block)
        
        acc += bias_ref[n].astype(jnp.float32)
        out_ref[m, n] = acc.astype(jnp.bfloat16)
    
    # Temporary buffer for matmul results
    temp_shape = jax.ShapeDtypeStruct((batch_size, out_features), jnp.bfloat16)
    
    # Compute matmul + bias
    matmul_out = pl.pallas_call(
        matmul_kernel,
        out_shape=temp_shape,
        grid=grid,
        in_specs=(
            pl.BlockSpec((batch_size, in_features), lambda m, n: (m, 0)),
            pl.BlockSpec((in_features, out_features), lambda m, n: (0, n)),
            pl.BlockSpec((out_features,), lambda m, n: (0,)),
        ),
        out_specs=pl.BlockSpec((batch_size, out_features), lambda m, n: (m, n)),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel")
        ),
    )(x, weight, bias)
    
    # Now apply softmax along axis 1
    def softmax_kernel(ref, out_ref):
        m = pl.program_id(0)
        
        # Read the entire row for softmax
        row = ref[m, :]
        
        # Compute softmax: exp(x - max(x)) / sum(exp(x - max(x)))
        max_val = jnp.max(row)
        exp_vals = jnp.exp(row.astype(jnp.float32) - max_val)
        sum_exp = jnp.sum(exp_vals)
        result = (exp_vals / sum_exp).astype(jnp.bfloat16)
        
        out_ref[m, :] = result
    
    softmax_grid = (batch_size // block_size,)
    
    return pl.pallas_call(
        softmax_kernel,
        out_shape=jax.ShapeDtypeStruct((batch_size, out_features), jnp.bfloat16),
        grid=softmax_grid,
        in_specs=(
            pl.BlockSpec((batch_size, out_features), lambda m: (m, 0)),
        ),
        out_specs=pl.BlockSpec((batch_size, out_features), lambda m: (m, 0)),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel",)
        ),
    )(matmul_out)
