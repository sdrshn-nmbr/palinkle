import jax
import jax.numpy as jnp
import jax.pallas as pl
import jax.pallas as plp
import jax.interpreters.tpu as pltpu

def workload(x, weight, bias, add_value):
    """TPU Pallas kernel for Matmul + Add + Swish + Tanh + GELU + Hardtanh."""
    
    block_size = 128  # Block size for tiling
    
    def kernel(ref_x, ref_weight, ref_bias, ref_add_value, ref_out):
        # Get program IDs for grid layout
        m_block = pl.program_id(0)
        n_block = pl.program_id(1)
        
        # Initialize accumulator in float32 for better precision
        acc = jnp.zeros((block_size, block_size), dtype=jnp.float32)
        
        # Matmul: x @ weight with accumulation in float32
        for k_block in range(x.shape[1] // block_size):
            x_block = ref_x[m_block * block_size:(m_block + 1) * block_size,
                           k_block * block_size:(k_block + 1) * block_size]
            weight_block = ref_weight[k_block * block_size:(k_block + 1) * block_size,
                                     n_block * block_size:(n_block + 1) * block_size]
            acc = acc + jnp.dot(x_block.astype(jnp.float32), 
                               weight_block.astype(jnp.float32))
        
        # Add bias
        bias_block = ref_bias[n_block * block_size:(n_block + 1) * block_size]
        acc = acc + bias_block[None, :]
        
        # Add add_value
        add_value_block = ref_add_value[n_block * block_size:(n_block + 1) * block_size]
        acc = acc + add_value_block[None, :]
        
        # Convert to bfloat16 for subsequent operations
        x_val = acc.astype(jnp.bfloat16)
        
        # Swish: x * sigmoid(x)
        x_val = x_val * jax.nn.sigmoid(x_val)
        
        # Tanh: tanh(x)
        x_val = jnp.tanh(x_val)
        
        # GELU: x * sigmoid(1.702 * x)
        x_val = x_val * jax.nn.sigmoid(1.702 * x_val)
        
        # Hardtanh: clip(x, -1, 1)
        x_val = jnp.clip(x_val, -1.0, 1.0)
        
        # Write output
        ref_out[m_block * block_size:(m_block + 1) * block_size,
                n_block * block_size:(n_block + 1) * block_size] = x_val
    
    # Grid dimensions
    grid_m = x.shape[0] // block_size
    grid_n = x.shape[1] // block_size
    
    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        grid=(grid_m, grid_n),
        in_specs=(
            pl.BlockSpec((grid_m, block_size, block_size), 
                        lambda m, n, bm, bn: (m * block_size, bm * block_size)),
            pl.BlockSpec((block_size, grid_n, block_size),
                        lambda m, n, bm, bn: (bm * block_size, n * block_size)),
            pl.BlockSpec((grid_n, block_size),
                        lambda m, n, bm, bn: (n * block_size,)),
            pl.BlockSpec((grid_n, block_size),
                        lambda m, n, bm, bn: (n * block_size,)),
        ),
        out_specs=pl.BlockSpec((block_size, block_size),
                              lambda m, n, bm, bn: (m * block_size, n * block_size)),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel")
        ),
    )(x, weight, bias, add_value)
