import jax
import jax.numpy as jnp
import pallas as pl
import pytpu

def workload(x, weight, bias):
    """Fused kernel: matmul + sum + max + mean + 2*logsumexp"""
    
    # After matmul, shape is [4096, 8192], after sum/max/mean/logsumexp it becomes [4096, 1]
    def kernel(x_ref, weight_ref, bias_ref, out_ref):
        # Matmul: x @ weight + bias
        # Use float32 for accumulation
        def matmul_body(i, j, k):
            return jnp.sum(x_ref[i, :] * weight_ref[:, j]) + bias_ref[j]
        
        # For simplicity, use JAX operations directly in the kernel
        # The PTLU compiler will lower this to TPU operations
        x_matmul = jnp.dot(x_ref, weight_ref)
        x_sum = jnp.sum(x_matmul, axis=1, keepdims=True)
        x_max = jnp.max(x_sum, axis=1, keepdims=True)
        x_mean = jnp.mean(x_max, axis=1, keepdims=True)
        x_lse1 = jax.scipy.special.logsumexp(x_mean, axis=1, keepdims=True)
        x_lse2 = jax.scipy.special.logsumexp(x_lse1, axis=1, keepdims=True)
        out_ref[:] = x_lse2
    
    # Grid: process all 4096 rows in parallel
    # Block size for matmul: use 128 for vectorization along K dimension
    block_size = 128
    
    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((4096, 1), jnp.float32),
        grid=(4096,),
        in_specs=(
            pl.BlockSpec((4096, block_size), lambda i: (i, 0)),  # x
            pl.BlockSpec((block_size, 8192), lambda i: (i % 8192, i // 8192)),  # weight
            pl.BlockSpec((8192,), lambda i: (i,)),  # bias
        ),
        out_specs=pl.BlockSpec((block_size,), lambda i: (i,)),
        compiler_params=pytpu.CompilerParams(
            dimension_semantics=("parallel",)
        ),
    )(x, weight, bias)
