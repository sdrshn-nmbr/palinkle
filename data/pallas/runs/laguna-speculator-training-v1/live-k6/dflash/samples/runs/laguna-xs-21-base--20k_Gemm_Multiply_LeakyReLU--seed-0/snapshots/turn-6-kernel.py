import jax
import jax.numpy as jnp
import jax.pallas as pl
import jax.pallas as plp
import jax.interpreters.pallas as pl_impl

def workload(x, weight, bias):
    """
    TPU Pallas kernel for Gemm + Multiply + LeakyReLU.
    
    Computes: leaky_relu((x @ weight + bias) * 2.0)
    """
    M, K = x.shape  # 4096, 8192
    K2, N = weight.shape  # 8192, 8192
    
    block_size = 128  # Block size for tiling
    
    def kernel(ref_x, ref_weight, ref_bias, ref_out):
        # Compute matmul result in float32 for accumulation
        def compute_matmul():
            # Accumulate in float32
            acc = pl.zeros((M, N), dtype=jnp.float32)
            
            # Tile over K dimension
            for k in range(0, K, block_size):
                k_end = min(k + block_size, K)
                # Load blocks
                x_block = ref_x[k:k_end, :].astype(jnp.float32)
                w_block = ref_weight[:, k:k_end].astype(jnp.float32)
                # Matmul and accumulate
                acc = acc + jnp.dot(x_block, w_block)
            
            return acc
        
        # Compute matmul + bias
        matmul_result = compute_matmul()
        # Add bias (broadcast along first dimension)
        result = matmul_result + ref_bias
        
        # Multiply by 2.0
        result = result * 2.0
        
        # LeakyReLU: where x >= 0, x else x * 0.1
        result = jnp.where(result >= 0, result, result * 0.1)
        
        # Store result as bfloat16
        ref_out[...] = result.astype(x.dtype)
    
    # Grid dimensions
    grid = (M, N)
    
    # Block specs for inputs
    in_specs = (
        pl.BlockSpec((M, K), lambda: (0, 0)),  # x
        pl.BlockSpec((K, N), lambda: (0, 0)),  # weight
        pl.BlockSpec((N,), lambda: (0,)),      # bias
    )
    
    out_specs = pl.BlockSpec((M, N), lambda: (0, 0))
    
    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((M, N), x.dtype),
        grid=grid,
        in_specs=in_specs,
        out_specs=out_specs,
        compiler_params=jax.pallas TPUCompilerParams(
            dimension_semantics=("parallel", "parallel")
        ),
    )(x, weight, bias)
