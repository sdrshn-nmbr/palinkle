import jax
import jax.numpy as jnp
import jax.pallas as pl
import jax.pallas as plp
import jax.interpreters.tpu as pltpu

def workload(x, weight, bias):
    """Matmul + Swish + Scaling kernel."""
    # Block size for tiling - use multiples of 8 for bf16
    block_m = 128
    block_k = 128
    block_n = 128
    
    # Grid dimensions
    grid_m = x.shape[0] // block_m
    grid_n = x.shape[1] // block_n
    
    def kernel(ref_x, ref_weight, ref_bias, ref_out):
        # Get program IDs
        m_start = pl.program_id(0) * block_m
        n_start = pl.program_id(1) * block_n
        
        # Accumulator in float32 for better precision
        acc = jnp.zeros((block_m, block_n), dtype=jnp.float32)
        
        # Tiled matmul over k dimension
        for k_start in range(0, x.shape[1], block_k):
            k_end = min(k_start + block_k, x.shape[1])
            actual_k = k_end - k_start
            
            # Load tiles
            x_tile = ref_x[m_start:m_start + block_m, k_start:k_end]
            w_tile = ref_weight[k_start:k_end, n_start:n_start + block_n]
            
            # Convert to float32 for accumulation
            x_f32 = x_tile.astype(jnp.float32)
            w_f32 = w_tile.astype(jnp.float32)
            
            # Matmul and accumulate
            acc += jnp.dot(x_f32, w_f32)
        
        # Add bias (broadcast along m dimension)
        bias_tile = ref_bias[n_start:n_start + block_n]
        acc = acc + bias_tile[None, :]
        
        # Convert back to bfloat16
        acc_bf16 = acc.astype(jnp.bfloat16)
        
        # Swish activation: x * sigmoid(x)
        swish_out = acc_bf16 * jax.nn.sigmoid(acc_bf16)
        
        # Scaling by 2.0
        result = swish_out * 2.0
        
        # Write output
        ref_out[...] = result
    
    # Define block specs
    in_specs = (
        pl.BlockSpec((block_m, block_k), lambda mi, ni: (mi * block_m, 0)),
        pl.BlockSpec((block_k, block_n), lambda mi, ni: (0, ni * block_n)),
        pl.BlockSpec((block_n,), lambda mi, ni: (ni * block_n,)),
    )
    
    out_specs = pl.BlockSpec((block_m, block_n), lambda mi, ni: (mi * block_m, ni * block_n))
    
    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        grid=(grid_m, grid_n),
        in_specs=in_specs,
        out_specs=out_specs,
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel")
        ),
    )(x, weight, bias)
