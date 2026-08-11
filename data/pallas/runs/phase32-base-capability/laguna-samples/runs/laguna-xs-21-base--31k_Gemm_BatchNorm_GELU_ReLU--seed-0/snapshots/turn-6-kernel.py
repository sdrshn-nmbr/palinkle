import jax
import jax.numpy as jnp
import pallas as pl
import pytensor as pltpu

def _batch_matmul_relu_gelu_bn_kernel(
    x_ref,
    gemm_weight_ref,
    gemm_bias_ref,
    bn_weight_ref,
    bn_bias_ref,
    eps_ref,
    out_ref,
):
    """Pallas kernel for Gemm + BatchNorm + GELU + ReLU."""
    # Get program IDs
    m_block = pl.program_id(0)
    n_block = pl.program_id(1)
    
    # Tile sizes
    TILE_M = 128
    TILE_N = 128
    TILE_K = 128
    
    # Accumulator for matmul result
    acc = jnp.zeros((TILE_M, TILE_N), dtype=jax.dtypes.bfloat16)
    
    # Compute matmul: x @ gemm_weight + gemm_bias
    for k_block in range(x_ref.shape[1] // TILE_K):
        # Load tiles
        x_tile = x_ref[m_block * TILE_M:(m_block + 1) * TILE_M, k_block * TILE_K:(k_block + 1) * TILE_K]
        w_tile = gemm_weight_ref[k_block * TILE_K:(k_block + 1) * TILE_K, n_block * TILE_N:(n_block + 1) * TILE_N]
        
        # Matmul accumulation
        acc = acc + jnp.dot(x_tile, w_tile)
    
    # Add bias
    acc = acc + gemm_bias_ref[n_block * TILE_N:(n_block + 1) * TILE_N]
    
    # Batch normalization
    # Compute mean and variance along batch dimension
    mean = jnp.mean(acc, axis=0, keepdims=True)
    var = jnp.mean((acc - mean) ** 2, axis=0, keepdims=True)
    
    # Normalize
    acc = (acc - mean) / jnp.sqrt(var + eps_ref[()])
    
    # Scale and shift
    acc = acc * bn_weight_ref[n_block * TILE_N:(n_block + 1) * TILE_N] + bn_bias_ref[n_block * TILE_N:(n_block + 1) * TILE_N]
    
    # GELU activation
    acc = jax.nn.gelu(acc)
    
    # ReLU activation
    acc = jax.nn.relu(acc)
    
    # Write output
    out_ref[m_block * TILE_M:(m_block + 1) * TILE_M, n_block * TILE_N:(n_block + 1) * TILE_N] = acc


def workload(x, gemm_weight, gemm_bias, bn_weight, bn_bias):
    """Workload: Gemm + BatchNorm + GELU + ReLU."""
    eps = 1e-5
    
    # Define grid based on output shape
    batch_size, in_features = x.shape
    out_features = gemm_weight.shape[1]
    
    # Use 128 as tile size for better vectorization
    TILE_M = 128
    TILE_N = 128
    
    grid = (batch_size // TILE_M, out_features // TILE_N)
    
    # Create specs for inputs
    x_spec = pl.BlockSpec((batch_size, in_features), lambda: (0, 0))
    gemm_weight_spec = pl.BlockSpec((in_features, out_features), lambda: (0, 0))
    gemm_bias_spec = pl.BlockSpec((out_features,), lambda: (0,))
    bn_weight_spec = pl.BlockSpec((out_features,), lambda: (0,))
    bn_bias_spec = pl.BlockSpec((out_features,), lambda: (0,))
    eps_spec = pl.BlockSpec((), lambda: ())
    
    out_spec = pl.BlockSpec((TILE_M, TILE_N), lambda i, j: (i * TILE_M, j * TILE_N))
    
    return pl.pallas_call(
        _batch_matmul_relu_gelu_bn_kernel,
        out_shape=jax.ShapeDtypeStruct((batch_size, out_features), dtype=x.dtype),
        grid=grid,
        in_specs=(
            x_spec,
            gemm_weight_spec,
            gemm_bias_spec,
            bn_weight_spec,
            bn_bias_spec,
            eps_spec,
        ),
        out_specs=out_spec,
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel", "parallel")),
    )(x, gemm_weight, gemm_bias, bn_weight, bn_bias, eps)
