import jax
import jax.numpy as jnp
import pallas as pl
import pytpu as pltpu


def gemm_kernel(
    A_ref,
    B_ref,
    C_ref,
    *,
    M=8192,
    N=28672,
    K=8192,
):
    # Block sizes for tiling
    block_M = 128
    block_N = 128
    block_K = 128

    # Get the program indices
    m_block = pl.program_id(0)
    n_block = pl.program_id(1)

    # Calculate the tile ranges
    m_start = m_block * block_M
    n_start = n_block * block_N

    # Initialize accumulator in float32
    acc = jnp.zeros((block_M, block_N), dtype=jnp.float32)

    # Tile over K dimension
    for k_block in range(K // block_K):
        # Load blocks from A and B
        a_tile = A_ref[
            m_start : m_start + block_M,
            k_block * block_K : (k_block + 1) * block_K,
        ]
        b_tile = B_ref[
            k_block * block_K : (k_block + 1) * block_K,
            n_start : n_start + block_N,
        ]

        # Compute dot product and accumulate
        acc = acc + jnp.dot(a_tile, b_tile)

    # Store result back to C in bfloat16
    C_ref[m_start : m_start + block_M, n_start : n_start + block_N] = acc.astype(
        jnp.bfloat16
    )


def workload(A, B):
    M, K = A.shape
    K, N = B.shape

    # Grid dimensions (number of blocks in each dimension)
    grid_M = (M + 127) // 128  # Round up for partial blocks
    grid_N = (N + 127) // 128

    return pl.pallas_call(
        gemm_kernel,
        out_shape=jnp.shape(A.value) if hasattr(A, "value") else A.shape,
        dtype=jnp.bfloat16,
        grid=(grid_M, grid_N),
        in_specs=(
            pl.BlockSpec((128, 128), lambda idx: (idx[0] * 128, slice(None))),
            pl.BlockSpec((128, 128), lambda idx: (slice(None), idx[1] * 128)),
        ),
        out_specs=pl.BlockSpec((128, 128), lambda idx: (idx[0] * 128, idx[1] * 128)),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel")
        ),
    )(A, B)
