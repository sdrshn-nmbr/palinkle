import jax
import jax.numpy as jnp
import jax.experimental.pallas as pl
import jax.experimental.pallas.tpu as pltpu

M = 8192
K = 8192
N = 28672

BLOCK_M = 128
BLOCK_N = 128
BLOCK_K = 128

def gemm_kernel(A_ref, B_ref, out_ref, scratch_ref):
    # Initialize float32 scratch accumulation
    scratch_ref[...] = jnp.zeros(scratch_ref.shape, dtype=jnp.float32)
    m, k = A_ref.shape
    k_chunk = BLOCK_K
    for k_idx in range(0, k, k_chunk):
        a = A_ref[:, k_idx:k_idx + k_chunk].astype(jnp.float32)
        b = B_ref[k_idx:k_idx + k_chunk, :].astype(jnp.float32)
        scratch_ref[...] = scratch_ref[...] + jnp.dot(a, b)
    out_ref[...] = scratch_ref[...].astype(jnp.bfloat16)

def workload(A, B):
    return pl.pallas_call(
        gemm_kernel,
        out_shape=jax.ShapeDtypeStruct((M, N), jnp.bfloat16),
        grid=(M // BLOCK_M, N // BLOCK_N),
        in_specs=(
            pl.BlockSpec((BLOCK_M, K), lambda i, j: (i * BLOCK_M, 0)),
            pl.BlockSpec((K, BLOCK_N), lambda i, j: (0, j * BLOCK_N)),
        ),
        out_specs=pl.BlockSpec((BLOCK_M, BLOCK_N), lambda i, j: (i * BLOCK_M, j * BLOCK_N)),
        scratch_shapes=(pltpu.VMEM((BLOCK_M, BLOCK_N), jnp.float32),),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel", "parallel")),
    )(A, B)
