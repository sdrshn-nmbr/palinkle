import jax
import jax.numpy as jnp
import pallas as pl
import pallas_core as plc
import jax.pallas as pl
import jax.pallas.lib as pllib
import jax.interpreters.pallas as pl
import jax.interpreters.pallas as pallas
import jax.interpreters.xla_extension as xla
from jax.interpreters import pallas as pallas_mod
import jax.numpy as jnp
from jax import lax
import jax

# Import TPU-specific modules
try:
    import jax.pallas.tpu as pltpu
except ImportError:
    # Fallback for different JAX versions
    import jax.interpreters.pallas as pl
    pl
    pltpu = type('pltpu', (), {})()
    # Create a mock module with necessary classes
    class MockCompilerParams:
        def __init__(self, dimension_semantics=None):
            self.dimension_semantics = dimension_semantics or ()
    
    class MockPrefetchScalarGridSpec:
        def __init__(self, num_scalar_prefetch, in_specs, out_specs, grid, scratch_shapes=None):
            self.num_scalar_prefetch = num_scalar_prefetch
            self.in_specs = in_specs
            self.out_specs = out_specs
            self.grid = grid
            self.scratch_shapes = scratch_shapes
    
    pltpu.CompilerParams = MockCompilerParams
    pltpu.PrefetchScalarGridSpec = MockPrefetchScalarGridSpec

def workload(x, weight, bias):
    """Compute: relu((matmul(x, weight) + bias - 2.0) * 1.5)"""
    
    # Output shape
    out_shape = (x.shape[0], weight.shape[1])
    
    # Tile size - use 128 for efficient TPU matmul
    TILE_M = 128  # batch dimension
    TILE_N = 128  # output dimension
    TILE_K = 128  # reduction dimension
    
    # Grid dimensions
    grid_m = (x.shape[0] + TILE_M - 1) // TILE_M
    grid_n = (out_shape[1] + TILE_N - 1) // TILE_N
    
    def kernel(ref_ref, weight_ref, bias_ref, out_ref):
        # Get program IDs
        m_idx = pl.program_id(0)
        n_idx = pl.program_id(1)
        
        # Compute tile boundaries
        m_start = m_idx * TILE_M
        n_start = n_idx * TILE_N
        
        m_end = min(m_start + TILE_M, x.shape[0])
        n_end = min(n_start + TILE_N, out_shape[1])
        
        # Accumulate in float32 for better precision
        acc = ref.zeros_like(out_ref.dtype, shape=(m_end - m_start, n_end - n_start))
        
        # Perform matmul with accumulation
        for k in range(0, weight.shape[1], TILE_K):
            k_end = min(k + TILE_K, weight.shape[1])
            
            # Load input tile
            x_tile = ref_ref[m_start:m_end, k:k+TILE_K] if k+TILE_K <= weight.shape[1] else ref_ref[m_start:m_end, k:]
            w_tile = weight_ref[k:TILE_K, n_start:n_end] if k+TILE_K <= weight.shape[0] else weight_ref[k:, n_start:n_end]
            
            # Matmul operation
            acc += jnp.dot(x_tile, w_tile)
        
        # Add bias
        bias_tile = bias_ref[n_start:n_end]
        acc = acc + bias_tile
        
        # Subtract 2.0
        acc = acc - 2.0
        
        # Multiply by 1.5
        acc = acc * 1.5
        
        # Apply ReLU
        acc = jnp.maximum(acc, 0.0)
        
        # Store result
        out_ref[m_start:m_end, n_start:n_end] = acc.astype(out_ref.dtype)
    
    # Use proper Pallas call with TPU compiler params
    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct(out_shape, jnp.bfloat16),
        grid=(grid_m, grid_n),
        in_specs=(
            pl.BlockSpec((TILE_M, weight.shape[1]), lambda i: (i[0] * TILE_M, 0)),
            pl.BlockSpec((weight.shape[0], TILE_N), lambda i: (0, i[1] * TILE_N)),
            pl.BlockSpec((weight.shape[1],), lambda i: (0,)),
        ),
        out_specs=pl.BlockSpec(out_shape, lambda i: (0, 0)),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel", "parallel")),
    )(x, weight, bias)
