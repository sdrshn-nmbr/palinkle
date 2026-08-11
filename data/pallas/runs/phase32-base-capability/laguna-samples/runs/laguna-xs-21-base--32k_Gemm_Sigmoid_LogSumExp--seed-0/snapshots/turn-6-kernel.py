import jax
import jax.numpy as jnp
import pallas as pl
import pallas.core as plc
import triton.language as tl


def gemm_sigmoid_logsumexp_kernel(
    x_ref,
    w1_ref,
    b1_ref,
    w2_ref,
    b2_ref,
    out_ref,
):
    """Pallas kernel for Gemm + Sigmoid + Gemm + LogSumExp."""
    # Get grid dimensions
    m = pl.program_id(0)  # batch dimension (16384)
    
    # Block size for matrix multiplication
    block_m = 128
    block_k = 128
    block_n = 128
    
    # First matmul: x @ w1.T + b1
    # x has shape [16384, 2048], w1 has shape [4096, 2048]
    # We compute x[m, :] @ w1.T to get [4096] output
    
    # Accumulate in float32 for better precision
    acc = jnp.zeros((4096,), dtype=jnp.float32)
    
    # Tile over the reduction dimension (2048)
    for k in range(0, 2048, block_k):
        x_block = x_ref[m, k : k + block_k].astype(jnp.float32)
        w1_block = w1_ref[:, k : k + block_k].astype(jnp.float32)
        acc = acc + jnp.sum(x_block[:, None, :] * w1_block[None, :, :], axis=0)
    
    # Add bias
    acc = acc + b1_ref[:].astype(jnp.float32)
    
    # Apply sigmoid
    acc = jax.nn.sigmoid(acc)
    
    # Second matmul: acc @ w2.T + b2
    # acc has shape [4096], w2 has shape [1024, 4096]
    # We compute acc @ w2.T to get [1024] output
    
    acc2 = jnp.zeros((1024,), dtype=jnp.float32)
    
    for k in range(0, 4096, block_n):
        acc_block = acc[k : k + block_n].astype(jnp.float32)
        w2_block = w2_ref[:, k : k + block_n].astype(jnp.float32)
        acc2 = acc2 + jnp.sum(acc_block[:, None] * w2_block[None, :, :], axis=0)
    
    # Add bias
    acc2 = acc2 + b2_ref[:].astype(jnp.float32)
    
    # Apply logsumexp
    result = jnp.log(jnp.sum(jnp.exp(acc2)))
    
    # Store output
    out_ref[m] = result.astype(x_ref.dtype)


def workload(x, w1, b1, w2, b2):
    """Main workload function."""
    # Grid is along batch dimension
    grid = (x.shape[0],)
    
    return pl.pallas_call(
        gemm_sigmoid_logsumexp_kernel,
        out_shape=jax.ShapeDtypeStruct((x.shape[0],), x.dtype),
        grid=grid,
        in_specs=(
            pl.BlockSpec((x.shape[0], x.shape[1]), lambda b: (b, slice(None))),
            pl.BlockSpec((w1.shape[0], w1.shape[1]), lambda b: (slice(None), b)),
            pl.BlockSpec((b1.shape[0],), lambda b: (slice(None),)),
            pl.BlockSpec((w2.shape[0], w2.shape[1]), lambda b: (slice(None), b)),
            pl.BlockSpec((b2.shape[0],), lambda b: (slice(None),)),
        ),
        out_specs=pl.BlockSpec((x.shape[0],), lambda b: (b,)),
        compiler_params=jax.tpu.CompilerParams(
            dimension_semantics=("parallel",)
        ),
    )(x, w1, b1, w2, b2)
