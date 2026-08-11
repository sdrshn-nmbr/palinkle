import jax
import jax.numpy as jnp
import jax.pallas as pl
import jax.pallas.lib as pllib
from jax.experimental import shard_map

def workload(x, weight, bias):
    """
    Gemm + Scaling + Hardtanh + GELU workload.
    
    Operations:
    1. y = matmul(x, weight) + bias
    2. y = y * 0.5
    3. y = clip(y, -2, 2)  # hardtanh with min=-2, max=2
    4. y = gelu_approx(y)  # 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    """
    batch_size = x.shape[0]
    out_features = weight.shape[1]
    
    # Define block sizes for TPU efficiency
    # For bfloat16 on TPU, use multiples of 8 for vector dimensions
    block_m = 128  # block size for batch dimension
    block_k = 128  # block size for reduction dimension
    block_n = 128  # block size for output dimension
    
    def pallas_kernel(
        x_ref,
        weight_ref,
        bias_ref,
        out_ref,
    ):
        # Get program indices
        m_idx = pl.program_id(0)  # batch block index
        n_idx = pl.program_id(1)  # output feature block index
        
        # Compute global indices
        m_start = m_idx * block_m
        n_start = n_idx * block_n
        
        # Initialize output with hardtanh bounds
        # Output shape for this block
        local_m = min(block_m, batch_size - m_start)
        local_n = min(block_n, out_features - n_start)
        
        # Accumulate in float32 for better precision
        acc = jnp.zeros((local_m, local_n), dtype=jnp.float32)
        
        # Perform GEMM: y = x @ weight + bias
        # We need to iterate over the reduction dimension (k)
        for k_idx in range(x.shape[1] // block_k):
            k_start = k_idx * block_k
            
            # Load x block (local_m x block_k)
            x_block = x_ref[m_start:m_start + local_m, k_start:k_start + block_k]
            
            # Load weight block (block_k x local_n)
            weight_block = weight_ref[k_start:k_start + block_k, n_start:n_start + local_n]
            
            # Accumulate
            acc = acc + (x_block.astype(jnp.float32) * weight_block.astype(jnp.float32))
        
        # Handle remaining k dimension if not divisible by block_k
        k_remaining = x.shape[1] % block_k
        if k_remaining > 0:
            x_block = x_ref[m_start:m_start + local_m, -k_remaining:]
            weight_block = weight_ref[-k_remaining:, n_start:n_start + local_n]
            acc = acc + (x_block.astype(jnp.float32) * weight_block.astype(jnp.float32))
        
        # Add bias
        bias_block = bias_ref[n_start:n_start + local_n]
        acc = acc + bias_block[None, :]
        
        # Scale by 0.5
        acc = acc * 0.5
        
        # Hardtanh: clip to [-2, 2]
        acc = jnp.clip(acc, -2.0, 2.0)
        
        # GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        # Using the approximation from the AST
        x_gelu = acc
        sqrt_2_over_pi = jnp.sqrt(2.0 / jnp.pi)
        gelu_term = sqrt_2_over_pi * (x_gelu + 0.044715 * jnp.pow(x_gelu, 3))
        acc = 0.5 * x_gelu * (1.0 + jnp.tanh(gelu_term))
        
        # Write output
        out_ref[m_start:m_start + local_m, n_start:n_start + local_n] = acc.astype(x.dtype)
    
    # Grid dimensions
    grid_m = (batch_size + block_m - 1) // block_m
    grid_n = (out_features + block_n - 1) // block_n
    
    # Define block specs
    x_block_shape = (block_m, block_k)
    weight_block_shape = (block_k, block_n)
    bias_block_shape = (block_n,)
    out_block_shape = (block_m, block_n)
    
    # For simplicity, use the same grid for all inputs
    # We'll handle the actual indexing inside the kernel
    return pl.pallas_call(
        pallas_kernel,
        out_shape=jnp.shape_struct((batch_size, out_features), x.dtype),
        grid=(grid_m, grid_n),
        in_specs=(
            pl.BlockSpec((batch_size, weight.shape[1]), lambda indices: (0, 0)),
            pl.BlockSpec(weight.shape, lambda indices: (0, 0)),
            pl.BlockSpec((out_features,), lambda indices: (0,)),
        ),
        out_specs=pl.BlockSpec((batch_size, out_features), lambda indices: (0, 0)),
        compiler_params=pllib.sharding.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            in_specs=(),
            out_specs=(),
            grid=(grid_m, grid_n),
        ),
    )(x, weight, bias)
