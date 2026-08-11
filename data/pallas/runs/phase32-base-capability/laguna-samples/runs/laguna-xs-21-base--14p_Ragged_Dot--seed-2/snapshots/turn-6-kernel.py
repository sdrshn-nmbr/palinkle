import jax
import jax.numpy as jnp
import pallas as pl
import pytpu as pltpu


def ragged_dot_kernel(
    x_ref,
    weights_ref,
    out_ref,
):
    """Kernel for grouped matmul (ragged dot).
    
    Performs einsum 'gmk,gkn->gmn' which is a grouped matrix multiplication.
    Each group (g) does an independent matmul.
    """
    # Get program IDs
    g = pl.program_id(0)  # group index
    m = pl.program_id(1)  # output row index within group
    n = pl.program_id(2)  # output column index
    
    # Accumulate in float32 for better precision
    acc = 0.0
    
    # k dimension is the reduction dimension
    for k in range(x_ref.shape[2]):
        acc += x_ref[m, k] * weights_ref[k, n]
    
    out_ref[m, n] = acc


def workload(x, weights):
    """Grouped matmul workload for Mixtral MoE.
    
    Args:
        x: Input tensor of shape [8, 1024, 4096] with dtype bfloat16
        weights: Weight tensor of shape [8, 4096, 14336] with dtype bfloat16
    
    Returns:
        Output tensor of shape [8, 1024, 14336] with dtype bfloat16
    """
    # Grid dimensions: 8 groups, 1024 rows, 14336 columns
    # For TPU, we need to tile appropriately
    # Block sizes should be multiples of 8 for bf16
    
    block_m = 128  # tile size for m dimension
    block_n = 128  # tile size for n dimension
    block_k = 8    # tile size for k dimension (reduction)
    
    # Grid spec: (num_groups, num_blocks_m, num_blocks_n)
    grid = (8, 1024 // block_m, 14336 // block_n)
    
    def kernel(ref_x, ref_weights, ref_out):
        # Get block indices
        g = pl.program_id(0)
        m_block = pl.program_id(1)
        n_block = pl.program_id(2)
        
        # Compute the starting indices for this block
        m_start = m_block * block_m
        n_start = n_block * block_n
        
        # Initialize output to zero
        ref_out[...] = 0.0
        
        # Perform the matmul for this block
        # For grouped matmul: out[g, m_start:m_start+block_m, n_start:n_start+block_n] 
        # = sum_k x[g, m_start:m_start+block_m, k] * weights[g, k, n_start:n_start+block_n]
        
        # Accumulate in float32
        acc = 0.0
        
        # Iterate over k dimension with tiling
        for k_block in range(ref_x.shape[2] // block_k):
            k_start = k_block * block_k
            
            # Load blocks from x and weights
            x_block = ref_x[g, m_start:m_start+block_m, k_start:k_start+block_k]
            w_block = ref_weights[g, k_start:k_start+block_k, n_start:n_start+block_n]
            
            # Accumulate
            acc += jnp.sum(x_block * w_block, axis=(1, 2))
        
        # Write result to output
        ref_out[g, m_start:m_start+block_m, n_start:n_start+block_n] = acc
    
    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((8, 1024, 14336), jnp.bfloat16),
        grid=grid,
        in_specs=(
            pl.BlockSpec((8, 1024, 4096), lambda g, m, n: (g, m, n)),
            pl.BlockSpec((8, 4096, 14336), lambda g, m, n: (g, m, n)),
        ),
        out_specs=pl.BlockSpec((8, 1024, 14336), lambda g, m, n: (g, m, n)),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel", "parallel")
        ),
    )(x, weights)
