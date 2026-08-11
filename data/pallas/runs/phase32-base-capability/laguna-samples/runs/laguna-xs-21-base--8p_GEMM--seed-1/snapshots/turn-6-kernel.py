import jax
import jax.numpy as jnp
import pallas as pl
import pl as pltpu

def gemm_kernel(A_ref, B_ref, C_ref):
    """Pallas kernel for matrix multiplication C = A @ B."""
    m, k = A_ref.shape
    k2, n = B_ref.shape
    assert k == k2, f"Inner dimensions must match: {k} != {k2}"
    
    # Initialize C to zeros in float32 for accumulation
    C_ref[...] = jnp.zeros((m, n), dtype=jnp.float32)
    
    # Perform the matrix multiplication using jnp.dot
    # This will be lowered to TPU matmul
    C_ref[...] = jnp.dot(A_ref, B_ref, precision=jax.lax.Precision.DEFAULT)

def workload(A, B):
    """Workload for dense matrix multiplication."""
    # Grid dimensions: tile along M and N dimensions
    # Block size: use 128 for vectorized dimensions (as per API docs)
    block_m = 128
    block_n = 128
    
    # For bf16, inner dimension K should be multiple of 8
    block_k = 8
    
    # Calculate grid size
    m, k = A.shape
    k2, n = B.shape
    
    # Grid along M and N dimensions
    grid_m = (m + block_m - 1) // block_m
    grid_n = (n + block_n - 1) // block_n
    
    # Define block specs for inputs and output
    def index_map_A(row, col):
        return (row * block_m, col * block_k)
    
    def index_map_B(row, col):
        return (row * block_k, col * block_n)
    
    def index_map_C(row, col):
        return (row * block_m, col * block_n)
    
    # Use BlockSpec for tiling
    in_specs = (
        pl.BlockSpec((block_m, block_k), lambda indices: (indices[0], indices[1])),
        pl.BlockSpec((block_k, block_n), lambda indices: (indices[0], indices[1])),
    )
    
    out_specs = pl.BlockSpec((block_m, block_n), lambda indices: (indices[0], indices[1]))
    
    # Call pallas_call with proper compiler params
    return pl.pallas_call(
        gemm_kernel,
        out_shape=jax.ShapeDtypeStruct((m, n), jnp.bfloat16),
        grid=(grid_m, grid_n),
        in_specs=in_specs,
        out_specs=out_specs,
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel")
        ),
    )(A, B)
