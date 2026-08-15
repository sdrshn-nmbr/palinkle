import jax
import jax.numpy as jnp
import jax.pallas as pl
import jax.pallas as plp
import jax.interpreters.tpu as pltpu


def gemm_kernel(
    A_ref,
    B_ref,
    C_ref,
    *,
    M: int,
    N: int,
    K: int,
):
    """Pallas kernel for GEMM: C = A @ B"""
    m = pl.program_id(0)
    n = pl.program_id(1)
    
    # Tile size for K dimension
    tile_k = 128
    
    # Accumulate in float32
    acc = jnp.zeros((), dtype=jnp.float32)
    
    for k_start in range(0, K, tile_k):
        k_end = min(k_start + tile_k, K)
        
        # Load A tile (M, tile_k)
        a_tile = A_ref[m, k_start:k_end].astype(jnp.float32)
        
        # Load B tile (tile_k, N)
        b_tile = B_ref[k_start:k_end, n].astype(jnp.float32)
        
        # Compute dot product for this tile
        acc = acc + jnp.sum(a_tile * b_tile)
    
    # Store result
    C_ref[m, n] = acc.astype(jnp.bfloat16)


def workload(A, B):
    """GEMM workload: C = A @ B for Llama-70B FFN dimensions"""
    M, K = A.shape  # (8192, 8192)
    K, N = B.shape  # (8192, 28672)
    
    block_m = 128
    block_n = 128
    
    grid = (M // block_m, N // block_n)
    
    return pl.pallas_call(
        gemm_kernel,
        out_shape=jax.ShapeDtypeStruct((M, N), jnp.bfloat16),
        grid=grid,
        in_specs=(
            pl.BlockSpec((M, K), lambda m, n: (m, 0)),
            pl.BlockSpec((K, N), lambda m, n: (0, n)),
        ),
        out_specs=pl.BlockSpec((M, N), lambda m, n: (m, n)),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel")
        ),
        interpret=False,
        M=M,
        N=N,
        K=K,
    )(A, B)
