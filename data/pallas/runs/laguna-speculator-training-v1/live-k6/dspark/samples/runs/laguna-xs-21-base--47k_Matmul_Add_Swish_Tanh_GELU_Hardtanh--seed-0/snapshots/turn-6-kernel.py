import jax
import jax.numpy as jnp
import jax.pallas as pl
import jax.pallas as plp
import jax.interpreters.pallas as pallas
from jax.interpreters import xla_bridge
from jax import xla

# TPU-specific imports
try:
    from jax.interpreters import pallas
    from jax.pallas import TPU
    pltpu = pallas
except ImportError:
    pltpu = None

def workload(x, weight, bias, add_value):
    """
    Implements: Matmul + Add + Swish + Tanh + GELU + Hardtanh
    
    x: [4096, 8192] bfloat16
    weight: [8192, 8192] bfloat16
    bias: [8192] bfloat16
    add_value: [8192] bfloat16
    output: [4096, 8192] bfloat16
    """
    
    # Output shape
    M, K = x.shape  # 4096, 8192
    _, N = weight.shape  # 8192
    
    # Block size for matmul tiling
    block_m = 128
    block_n = 128
    block_k = 128
    
    def kernel(ref_x, ref_weight, ref_bias, ref_add_value, ref_out):
        # Get program indices
        m_block = pl.program_id(0)
        n_block = pl.program_id(1)
        
        # Compute tile offsets
        m_start = m_block * block_m
        n_start = n_block * block_n
        
        # Accumulator in float32 for better precision
        acc = jnp.zeros((block_m, block_n), dtype=jnp.float32)
        
        # Matmul kernel: accumulate over K dimension
        for k_block in range(K // block_k):
            k_start = k_block * block_k
            
            # Load tiles
            x_tile = ref_x[m_start:m_start + block_m, k_start:k_start + block_k]
            w_tile = ref_weight[k_start:k_start + block_k, n_start:n_start + block_n]
            
            # Convert to float32 for accumulation
            x_f32 = x_tile.astype(jnp.float32)
            w_f32 = w_tile.astype(jnp.float32)
            
            # Matmul and accumulate
            acc = acc + jnp.dot(x_f32, w_f32)
        
        # Add bias (broadcast along M dimension)
        bias_tile = ref_bias[n_start:n_start + block_n]
        acc = acc + bias_tile[None, :]
        
        # Add add_value (broadcast along M dimension)
        add_tile = ref_add_value[n_start:n_start + block_n]
        acc = acc + add_tile[None, :]
        
        # Convert back to bfloat16 for activation functions
        acc_bf16 = acc.astype(jnp.bfloat16)
        
        # Swish: x * sigmoid(x)
        acc_bf16 = acc_bf16 * jax.nn.sigmoid(acc_bf16)
        
        # Tanh
        acc_bf16 = jnp.tanh(acc_bf16)
        
        # GELU: approximate using sigmoid
        acc_bf16 = acc_bf16 * jax.nn.sigmoid(1.7159 * acc_bf16)
        
        # Hardtanh: clip to [-1, 1]
        acc_bf16 = jnp.clip(acc_bf16, -1.0, 1.0)
        
        # Write output
        ref_out[m_start:m_start + block_m, n_start:n_start + block_n] = acc_bf16
    
    # Grid dimensions
    grid_m = M // block_m
    grid_n = N // block_n
    
    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((M, N), x.dtype),
        grid=(grid_m, grid_n),
        in_specs=(
            pl.BlockSpec((block_m, block_k), lambda m, n, k: (m * block_m, k * block_k)),
            pl.BlockSpec((block_k, block_n), lambda m, n, k: (k * block_k, n * block_n)),
            pl.BlockSpec((block_n,), lambda m, n, k: (n * block_n,)),
            pl.BlockSpec((block_n,), lambda m, n, k: (n * block_n,)),
        ),
        out_specs=pl.BlockSpec((block_m, block_n), lambda m, n, k: (m * block_m, n * block_n)),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel")
        ),
    )(x, weight, bias, add_value)
