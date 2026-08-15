import jax
import jax.numpy as jnp
import jax.pallas as pl
import jax.pallas as plp
import jax.interpreters.tpu as pltpu

def workload(x, weight, bias):
    """Gemm + Scaling + Hardtanh + GELU kernel."""
    
    # Block size for TPU matmul - use multiples of 8 for bf16
    block_m = 128
    block_n = 128
    block_k = 8
    
    # Grid dimensions
    grid_m = x.shape[0] // block_m
    grid_n = x.shape[1] // block_n
    
    def kernel(ref_x, ref_weight, ref_bias, out_ref):
        # Get program IDs
        m_start = pl.program_id(0) * block_m
        n_start = pl.program_id(1) * block_n
        
        # Initialize accumulator in float32
        acc = jnp.zeros((block_m, block_n), dtype=jnp.float32)
        
        # Matmul with accumulation
        for k_start in range(0, x.shape[1], block_k):
            # Load blocks
            x_block = ref_x[m_start:m_start + block_m, k_start:k_start + block_k]
            w_block = ref_weight[k_start:k_start + block_k, n_start:n_start + block_n]
            
            # Accumulate in float32
            acc = acc + jnp.dot(x_block.astype(jnp.float32), w_block.astype(jnp.float32))
        
        # Add bias
        bias_block = ref_bias[n_start:n_start + block_n]
        acc = acc + bias_block
        
        # Scale by 0.5
        acc = acc * 0.5
        
        # Hardtanh: clip to [-2, 2]
        acc = jnp.clip(acc, -2.0, 2.0)
        
        # GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        sqrt_2_pi = jnp.sqrt(2.0 / jnp.pi)
        gelu = 0.5 * acc * (1.0 + jnp.tanh(sqrt_2_pi * (acc + 0.044715 * jnp.power(acc, 3.0))))
        
        # Write output
        out_ref[m_start:m_start + block_m, n_start:n_start + block_n] = gelu.astype(jnp.bfloat16)
    
    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        grid=(grid_m, grid_n),
        in_specs=(
            pl.BlockSpec((block_m, block_k), lambda mi, ki: (mi * block_m, ki * block_k)),
            pl.BlockSpec((block_k, block_n), lambda ki, ni: (ki * block_k, ni * block_n)),
            pl.BlockSpec((block_n,), lambda ni: (ni * block_n,)),
        ),
        out_specs=pl.BlockSpec((block_m, block_n), lambda mi, ni: (mi * block_m, ni * block_n)),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel")
        ),
    )(x, weight, bias)
