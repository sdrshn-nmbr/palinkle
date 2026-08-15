import jax
import jax.numpy as jnp
import jax.pallas as pl
import jax.pallas.lib as pllib
import jax.pallas.tpu as pltpu


def workload(x, weight, bias):
    """TPU Pallas kernel for Matmul + Scale + ResidualAdd + Clamp + LogSumExp + Mish."""
    
    # Block size for matmul tiling
    block_m = 128  # Multiple of 8 for bf16
    block_n = 128  # Multiple of 8 for bf16
    block_k = 128  # Multiple of 8 for bf16
    
    # Grid dimensions
    grid_m = x.shape[0] // block_m
    grid_n = weight.shape[1] // block_n
    grid_k = weight.shape[0] // block_k
    
    def kernel(matmul_out_ref, x_ref, weight_ref, bias_ref, out_ref):
        # Matmul: x @ weight + bias
        # Accumulate in float32 for better precision
        acc = jnp.zeros((block_m, block_n), dtype=jnp.float32)
        
        for k_block in range(grid_k):
            x_block = x_ref[k_block * block_k:(k_block + 1) * block_k].astype(jnp.float32)
            w_block = weight_ref[:, k_block * block_k:(k_block + 1) * block_k].astype(jnp.float32)
            acc = acc + jnp.dot(x_block, w_block)
        
        # Add bias (broadcast along last dimension)
        acc = acc + bias_ref.astype(jnp.float32)
        
        # Scale by 2.0
        acc = acc * 2.0
        
        # ResidualAdd: add to itself (x + x = 2x, but we already scaled by 2)
        # Actually, the AST shows: x = x + x, which doubles the value
        # Since we already multiplied by 2.0, this would give us 4x
        # But looking at the AST more carefully:
        # 1. x = x @ weight + bias
        # 2. x = x * 2.0
        # 3. x = x + x  (residual add)
        acc = acc + acc
        
        # Clamp to [-10.0, 10.0]
        acc = jnp.clip(acc, -10.0, 10.0)
        
        # LogSumExp along axis 1 with keepdims
        # logsumexp(x, axis=1, keepdims=True)
        acc = jax.scipy.special.logsumexp(acc, axis=1, keepdims=True)
        
        # Mish: x * tanh(logaddexp(x, 0.0))
        softplus_x = jnp.logaddexp(acc, 0.0)
        mish_x = acc * jnp.tanh(softplus_x)
        
        # Final multiply by mish_x
        acc = acc * mish_x
        
        # Write output
        out_ref[...] = acc.astype(jnp.bfloat16)
    
    # Output shape
    out_shape = jax.ShapeDtypeStruct((x.shape[0], 1), jnp.bfloat16)
    
    # For the kernel, we need to handle the full matmul
    # Let's use a simpler approach with a single kernel that handles everything
    
    def full_kernel(x_ref, weight_ref, bias_ref, out_ref):
        # Get block indices
        m_block = pl.program_id(0)
        n_block = pl.program_id(1)
        
        # Matmul: compute x[m_block] @ weight[:, n_block]
        # For simplicity, let's compute the full matmul in the kernel
        
        # Read input blocks
        x_local = x_ref[...]
        weight_local = weight_ref[...]
        bias_local = bias_ref[...]
        
        # Matmul: x @ weight
        # Shape: [4096, 8192] @ [8192, 8192] -> [4096, 8192]
        # But output is [4096, 1], so we need to reduce along axis 1
        
        # Actually, let me re-read the spec...
        # The output is [4096, 1], and we do logsumexp along axis 1
        # So the matmul output should be [4096, 8192]
        
        # Let's compute the full matmul
        # For TPU efficiency, we should tile this
        
        # For now, let's use a simpler grid
        pass
    
    # Let me implement this more carefully
    # The output is [4096, 1], so we need to process all rows
    
    # Use a grid that processes all rows
    grid = (x.shape[0],)
    
    def kernel_single_row(x_ref, weight_ref, bias_ref, out_ref):
        row_idx = pl.program_id(0)
        
        # Get the row of x
        x_row = x_ref[row_idx, :]  # [8192]
        
        # Matmul: x_row @ weight + bias
        # x_row: [8192], weight: [8192, 8192], result: [8192]
        # Then add bias: [8192]
        result = jnp.dot(x_row, weight_ref).astype(jnp.float32) + bias_ref.astype(jnp.float32)
        
        # Scale by 2.0
        result = result * 2.0
        
        # ResidualAdd: add to itself
        result = result + result
        
        # Clamp to [-10.0, 10.0]
        result = jnp.clip(result, -10.0, 10.0)
        
        # LogSumExp along axis 0 (since result is 1D)
        # But we need to reshape for logsumexp with keepdims
        result_2d = result.reshape(1, -1)  # [1, 8192]
        result = jax.scipy.special.logsumexp(result_2d, axis=1, keepdims=True).reshape(-1)  # [1]
        
        # Mish
        softplus_x = jnp.logaddexp(result, 0.0)
        mish_x = result * jnp.tanh(softplus_x)
        
        # Final multiply
        result = result * mish_x
        
        out_ref[...] = result.astype(jnp.bfloat16)
    
    return pl.pallas_call(
        kernel_single_row,
        out_shape=out_shape,
        grid=grid,
        in_specs=(
            pl.BlockSpec((x.shape[1],), lambda row_idx: (row_idx,)),
            pl.BlockSpec((weight.shape[0], weight.shape[1]), lambda row_idx: (slice(None), slice(None))),
            pl.BlockSpec((bias.shape[0],), lambda row_idx: (slice(None),)),
        ),
        out_specs=pl.BlockSpec((1,), lambda row_idx: (row_idx,)),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel",)
        ),
    )(x, weight, bias)
