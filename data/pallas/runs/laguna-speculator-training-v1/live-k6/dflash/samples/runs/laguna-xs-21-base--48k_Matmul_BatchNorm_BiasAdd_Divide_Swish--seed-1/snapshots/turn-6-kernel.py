import jax
import jax.numpy as jnp
import jax.pallas as pl
import jax.pallas as plp
import jax.interpreters.tpu as pltpu

def workload(x, weight, linear_bias, bn_scale, bn_bias, bn_mean, bn_var, bias):
    """Matmul + BatchNorm + BiasAdd + Divide + Swish kernel."""
    
    bn_eps = 1e-05
    divide_value = 1.0
    
    # Output shape
    out_shape = jax.ShapeDtypeStruct(x.shape, x.dtype)
    
    # Block size for tiling - use multiples of 8 for bf16
    block_m = 128  # rows
    block_k = 128  # reduction dimension
    block_n = 128  # columns
    
    # Grid dimensions
    grid_m = (x.shape[0] + block_m - 1) // block_m
    grid_n = (x.shape[1] + block_n - 1) // block_n
    grid_k = (x.shape[1] + block_k - 1) // block_k
    
    def kernel(ref_out, x_ref, weight_ref, linear_bias_ref, 
               bn_scale_ref, bn_bias_ref, bn_mean_ref, bn_var_ref, bias_ref):
        # Get program IDs
        m_idx = pl.program_id(0)
        n_idx = pl.program_id(1)
        
        # Compute tile bounds
        m_start = m_idx * block_m
        n_start = n_idx * block_n
        
        # Initialize accumulator in float32 for better precision
        acc = jnp.zeros((block_m, block_n), dtype=jnp.float32)
        
        # Matmul kernel: x @ weight
        for k_idx in range(grid_k):
            # Load tiles from x and weight
            x_tile = x_ref[
                m_start:min(m_start + block_m, x.shape[0]),
                k_idx * block_k:min((k_idx + 1) * block_k, x.shape[1])
            ]
            w_tile = weight_ref[
                k_idx * block_k:min((k_idx + 1) * block_k, weight.shape[0]),
                n_start:min(n_start + block_n, weight.shape[1])
            ]
            
            # Convert to float32 for accumulation
            x_f32 = x_tile.astype(jnp.float32)
            w_f32 = w_tile.astype(jnp.float32)
            
            # Accumulate
            acc = acc + jnp.dot(x_f32, w_f32)
        
        # Add linear bias
        bias_tile = linear_bias_ref[n_start:min(n_start + block_n, linear_bias_ref.shape[0])]
        acc = acc + bias_tile.astype(jnp.float32)
        
        # BatchNorm: (x - mean) / sqrt(var + eps) * scale + bias
        mean_tile = bn_mean_ref[n_start:min(n_start + block_n, bn_mean_ref.shape[0])]
        var_tile = bn_var_ref[n_start:min(n_start + block_n, bn_var_ref.shape[0])]
        scale_tile = bn_scale_ref[n_start:min(n_start + block_n, bn_scale_ref.shape[0])]
        bn_bias_tile = bn_bias_ref[n_start:min(n_start + block_n, bn_bias_ref.shape[0])]
        
        x_normalized = (acc - mean_tile.astype(jnp.float32)) / jnp.sqrt(var_tile.astype(jnp.float32) + bn_eps)
        acc = x_normalized * scale_tile.astype(jnp.float32) + bn_bias_tile.astype(jnp.float32)
        
        # Add bias
        bias_val = bias_ref[0]
        acc = acc + bias_val.astype(jnp.float32)
        
        # Divide
        acc = acc / divide_value
        
        # Swish: x * sigmoid(x)
        acc = acc * jax.nn.sigmoid(acc)
        
        # Write output
        ref_out[
            m_start:min(m_start + block_m, x.shape[0]),
            n_start:min(n_start + block_n, x.shape[1])
        ] = acc.astype(x.dtype)
    
    return pl.pallas_call(
        kernel,
        out_shape=out_shape,
        grid=(grid_m, grid_n),
        in_specs=(
            pl.no_block_spec,  # ref_out
            pl.BlockSpec((block_m, block_k), lambda m, n: (m * block_m, 0)),  # x_ref
            pl.BlockSpec((block_k, block_n), lambda m, n: (0, n * block_n)),  # weight_ref
            pl.BlockSpec((block_n,), lambda m, n: (n * block_n,)),  # linear_bias_ref
            pl.BlockSpec((block_n,), lambda m, n: (n * block_n,)),  # bn_scale_ref
            pl.BlockSpec((block_n,), lambda m, n: (n * block_n,)),  # bn_bias_ref
            pl.BlockSpec((block_n,), lambda m, n: (n * block_n,)),  # bn_mean_ref
            pl.BlockSpec((block_n,), lambda m, n: (n * block_n,)),  # bn_var_ref
            pl.BlockSpec((1,), lambda m, n: (0,)),  # bias_ref
        ),
        out_specs=pl.no_block_spec,
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel")
        ),
    )(x, weight, linear_bias, bn_scale, bn_bias, bn_mean, bn_var, bias)
