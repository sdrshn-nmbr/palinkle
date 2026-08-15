import jax
import jax.numpy as jnp
import jax.pallas as pl
import jax.pallas as plp
import jax

def workload(x, weights):
    """Grouped matmul (ragged dot) for MoE - Mixtral 8x7B.
    
    Computes einsum("gmk,gkn->gmn", x, weights) which is equivalent to
    grouped matrix multiplication where each group g does an independent matmul.
    
    Args:
        x: Input tensor of shape [8, 1024, 4096] with dtype bfloat16
        weights: Weight tensor of shape [8, 4096, 14336] with dtype bfloat16
    
    Returns:
        Output tensor of shape [8, 1024, 14336] with dtype bfloat16
    """
    # Constants from the configuration
    num_groups = 8
    M = 1024  # x.shape[1]
    K = 4096  # x.shape[2] = weights.shape[1]
    N = 14336  # weights.shape[2]
    
    # Block sizes - TPU requires multiples of 8 for bf16 and 128 for vectorized dims
    block_m = 128
    block_n = 128
    block_k = 128
    
    def matmul_kernel(x_ref, w_ref, out_ref):
        """Pallas kernel for grouped matmul."""
        # Get the group index
        g = pl.program_id(0)
        
        # Get block indices for M and N dimensions
        m_block = pl.program_id(1)
        n_block = pl.program_id(2)
        
        # Calculate the starting positions
        m_start = m_block * block_m
        n_start = n_block * block_n
        
        # Initialize output block to zero in float32 for accumulation
        pl.when(m_start < M and n_start < N, lambda: out_ref[...].set(jnp.zeros((block_m, block_n), dtype=jnp.float32)))
        
        # Accumulate over K dimension
        for k_block in range(K // block_k):
            k_start = k_block * block_k
            
            # Load input blocks
            x_block = x_ref[m_start:m_start + block_m, k_start:k_start + block_k]
            w_block = w_ref[k_start:k_start + block_k, n_start:n_start + block_n]
            
            # Compute partial matmul and accumulate
            partial = jnp.dot(x_block, w_block)
            out_ref[m_start:m_start + block_m, n_start:n_start + block_n] = (
                out_ref[m_start:m_start + block_m, n_start:n_start + block_n] + partial
            )
    
    # Grid dimensions: (num_groups, num_m_blocks, num_n_blocks)
    num_m_blocks = (M + block_m - 1) // block_m
    num_n_blocks = (N + block_n - 1) // block_n
    grid = (num_groups, num_m_blocks, num_n_blocks)
    
    # Block specs for inputs and outputs
    x_spec = pl.BlockSpec(
        (num_groups, M, K),
        lambda g, m_block, n_block: (g, m_block * block_m, 0)
    )
    w_spec = pl.BlockSpec(
        (num_groups, K, N),
        lambda g, m_block, n_block: (g, 0, n_block * block_n)
    )
    out_spec = pl.BlockSpec(
        (num_groups, M, N),
        lambda g, m_block, n_block: (g, m_block * block_m, n_block * block_n)
    )
    
    return pl.pallas_call(
        matmul_kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        grid=grid,
        in_specs=(x_spec, w_spec),
        out_specs=out_spec,
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel", "parallel")
        ),
    )(x, weights)
