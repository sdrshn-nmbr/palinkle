import jax
import jax.numpy as jnp
import jax.pallas as pl
import jax.pallas as plp
import jax.interpreters.pallas as pallas
from jax.interpreters import pallas as pl
from jax.pallas.lib import SlabAllocator
import jax.numpy as jnp
from functools import partial

# TPU-specific imports
try:
    from jax.interpreters import xla_extension
    from jax.interpreters.pallas import TPUCompilerParams
    pltpu = xla_extension
except ImportError:
    # Fallback for non-TPU environments
    import jax.pallas as pltpu

def workload(x, weight, bias):
    """
    Gemm + Scaling + Hardtanh + GELU kernel.
    
    Operations:
    1. x = x @ weight + bias
    2. x = x * 0.5
    3. x = clip(x, -2, 2)  # Hardtanh
    4. x = x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))  # GELU
    """
    
    batch_size = x.shape[0]  # 4096
    in_features = x.shape[1]  # 8192
    out_features = weight.shape[1]  # 8192
    
    # Block size for TPU - must be multiple of 8 for bf16
    block_size = 128
    
    def gemm_scaling_hardtanh_gelu_kernel(
        x_ref, weight_ref, bias_ref, out_ref
    ):
        # Get program IDs for grid layout
        m_block = pl.program_id(0)  # batch dimension block
        n_block = pl.program_id(1)  # output dimension block
        
        # Compute tile offsets
        m_start = m_block * block_size
        n_start = n_block * block_size
        
        # Accumulator for matmul result in float32
        accumulator = jnp.zeros((block_size, block_size), dtype=jnp.float32)
        
        # Perform tiled matmul
        for k_block in range(in_features // block_size):
            k_start = k_block * block_size
            
            # Load tiles from x and weight
            x_tile = x_ref[m_start:m_start + block_size, k_start:k_start + block_size]
            w_tile = weight_ref[k_start:k_start + block_size, n_start:n_start + block_size]
            
            # Convert to float32 for accumulation
            x_tile_f32 = x_tile.astype(jnp.float32)
            w_tile_f32 = w_tile.astype(jnp.float32)
            
            # Accumulate
            accumulator = accumulator + jnp.dot(x_tile_f32, w_tile_f32)
        
        # Add bias (broadcast along batch dimension)
        bias_tile = bias_ref[n_start:n_start + block_size]
        accumulator = accumulator + bias_tile[None, :]
        
        # Convert back to bf16 for scaling
        result = accumulator.astype(jnp.bfloat16)
        
        # Scaling: x * 0.5
        result = result * 0.5
        
        # Hardtanh: clip(x, -2, 2)
        result = jnp.clip(result, -2.0, 2.0)
        
        # GELU: x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        # Compute the GELU approximation
        pi = jnp.pi
        sqrt_2_pi = jnp.sqrt(2.0 / pi)
        gelu_term = sqrt_2_pi * (result + 0.044715 * jnp.power(result, 3))
        result = result * 0.5 * (1.0 + jnp.tanh(gelu_term))
        
        # Write output
        out_ref[m_start:m_start + block_size, n_start:n_start + block_size] = result
    
    # Grid dimensions
    num_m_blocks = batch_size // block_size
    num_n_blocks = out_features // block_size
    
    # Create block specs
    def x_index_map(m_block, n_block):
        return (m_block * block_size, 0)
    
    def weight_index_map(m_block, n_block):
        return (0, n_block * block_size)
    
    def bias_index_map(m_block, n_block):
        return (n_block * block_size,)
    
    def out_index_map(m_block, n_block):
        return (m_block * block_size, n_block * block_size)
    
    # Use TPU compiler params
    compiler_params = pltpu.CompilerParams(
        dimension_semantics=("parallel", "parallel")
    )
    
    return pl.pallas_call(
        gemm_scaling_hardtanh_gelu_kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        grid=(num_m_blocks, num_n_blocks),
        in_specs=(
            pl.BlockSpec((block_size, block_size), x_index_map),
            pl.BlockSpec((block_size, block_size), weight_index_map),
            pl.BlockSpec((block_size,), bias_index_map),
        ),
        out_specs=pl.BlockSpec((block_size, block_size), out_index_map),
        compiler_params=compiler_params,
    )(x, weight, bias)
