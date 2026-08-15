import jax
import jax.numpy as jnp
import pallas as pl
import jax.pallas as pltpu


def matmul_kernel(ref_a, ref_b, ref_c):
    """Pallas kernel for matrix multiplication C = A @ B."""
    m = ref_a.shape[0]
    k = ref_a.shape[1]
    n = ref_b.shape[1]
    
    # Initialize output to zero
    ref_c[...] = jnp.zeros((m, n), dtype=jnp.float32)
    
    # Perform matrix multiplication using jnp.dot
    # The result is accumulated in float32
    result = jnp.dot(ref_a[...], ref_b[...])
    ref_c[...] = result


def workload(A, B):
    """Compute C = A @ B using Pallas kernel."""
    # Block size for tiling - use multiples of 8 for bf16
    block_m = 128
    block_k = 128
    block_n = 128
    
    # Grid dimensions
    grid_m = A.shape[0] // block_m
    grid_n = B.shape[1] // block_n
    
    def kernel(ref_a, ref_b, ref_c):
        # Get program IDs
        m_start = pl.program_id(0) * block_m
        n_start = pl.program_id(1) * block_n
        
        # Extract blocks
        a_block = ref_a[m_start:m_start + block_m, :]
        b_block = ref_b[:, n_start:n_start + block_n]
        
        # Compute partial result
        c_block = jnp.dot(a_block, b_block)
        
        # Write to output
        ref_c[m_start:m_start + block_m, n_start:n_start + block_n] = c_block
    
    # Define output shape
    out_shape = jax.ShapeDtypeStruct(A.shape, A.dtype)
    
    # Define block specs
    in_spec_a = pl.BlockSpec(
        (block_m, A.shape[1]),
        lambda mi, ni: (mi * block_m, 0)
    )
    in_spec_b = pl.BlockSpec(
        (A.shape[1], block_n),
        lambda mi, ni: (0, ni * block_n)
    )
    out_spec = pl.BlockSpec(
        (block_m, block_n),
        lambda mi, ni: (mi * block_m, ni * block_n)
    )
    
    # Create the Pallas call
    return pl.pallas_call(
        kernel,
        out_shape=out_shape,
        grid=(grid_m, grid_n),
        in_specs=(in_spec_a, in_spec_b),
        out_specs=(out_spec,),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel")
        ),
    )(A, B)
