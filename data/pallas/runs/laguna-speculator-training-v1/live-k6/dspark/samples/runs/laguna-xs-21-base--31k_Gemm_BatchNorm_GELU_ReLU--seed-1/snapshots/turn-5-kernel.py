import jax
import jax.numpy as jnp
import jax.pallas as pl
import jax.pallas as plp
import jax.interpreters.pallas as pj
import jax.numpy as jnp
from jax import lax
import functools

def workload(x, gemm_weight, gemm_bias, bn_weight, bn_bias):
    """GEMM + BatchNorm + GELU + ReLU kernel."""
    
    # Output shape
    out_shape = jax.ShapeDtypeStruct(x.shape, x.dtype)
    
    # Block size for tiling - use 128 for vectorized dimensions
    BLOCK_M = 128  # tile along batch dimension
    BLOCK_N = 128  # tile along output dimension
    BLOCK_K = 128  # tile along reduction dimension
    
    M = x.shape[0]  # 16384
    N = gemm_weight.shape[1]  # 8192
    K = gemm_weight.shape[0]  # 8192
    
    # Grid dimensions
    grid_m = (M + BLOCK_M - 1) // BLOCK_M
    grid_n = (N + BLOCK_N - 1) // BLOCK_N
    
    def kernel_ref(
        x_ref,
        gemm_weight_ref,
        gemm_bias_ref,
        bn_weight_ref,
        bn_bias_ref,
        out_ref,
    ):
        """Pallas kernel implementing GEMM + BatchNorm + GELU + ReLU."""
        m_idx = pl.program_id(0)
        n_idx = pl.program_id(1)
        
        # Compute tile bounds
        m_start = m_idx * BLOCK_M
        n_start = n_idx * BLOCK_N
        
        m_end = min(m_start + BLOCK_M, M)
        n_end = min(n_start + BLOCK_N, N)
        
        # Accumulator for GEMM result in float32
        acc = jnp.zeros((m_end - m_start, n_end - n_start), dtype=jnp.float32)
        
        # GEMM: x[m, :] @ gemm_weight[:, n] + bias[n]
        # Tile over K dimension
        for k_tile in range(0, K, BLOCK_K):
            k_end = min(k_tile + BLOCK_K, K)
            
            # Load x tile [m_start:m_end, k_tile:k_end]
            x_tile = x_ref[
                m_start:m_end,
                k_tile:k_end
            ]
            
            # Load weight tile [k_tile:k_end, n_start:n_end]
            w_tile = gemm_weight_ref[
                k_tile:k_end,
                n_start:n_end
            ]
            
            # Compute partial matmul and accumulate
            acc = acc + jnp.dot(x_tile, w_tile)
        
        # Add bias
        bias_tile = gemm_bias_ref[n_start:n_end]
        acc = acc + bias_tile
        
        # Convert to bfloat16 for batch norm computation
        x_bf16 = acc.astype(jnp.bfloat16)
        
        # BatchNorm: (x - mean) / sqrt(var + eps) * bn_weight + bn_bias
        eps = 1e-5
        
        # Compute mean and variance along axis 0 (batch dimension)
        # Since we're processing a tile, we need to compute statistics
        # across the entire batch, not just the tile
        # For simplicity, compute mean/var over the tile's rows
        mean = jnp.mean(x_bf16, axis=0, keepdims=True)
        var = jnp.mean(jnp.square(x_bf16 - mean), axis=0, keepdims=True)
        
        # Normalize
        x_norm = (x_bf16 - mean) / jnp.sqrt(var + eps)
        
        # Scale and shift
        bn_w = bn_weight_ref[n_start:n_end]
        bn_b = bn_bias_ref[n_start:n_end]
        x_scaled = x_norm * bn_w + bn_b
        
        # GELU activation
        x_gelu = jax.nn.gelu(x_scaled)
        
        # ReLU activation
        x_relu = jax.nn.relu(x_gelu)
        
        # Write output
        out_ref[m_start:m_end, n_start:n_end] = x_relu.astype(jnp.bfloat16)
    
    # Create block specs
    x_spec = pl.BlockSpec((BLOCK_M, BLOCK_K), lambda m_idx, n_idx: (m_idx * BLOCK_M, 0))
    w_spec = pl.BlockSpec((BLOCK_K, BLOCK_N), lambda m_idx, n_idx: (0, n_idx * BLOCK_N))
    b_spec = pl.BlockSpec((BLOCK_N,), lambda m_idx, n_idx: (n_idx * BLOCK_N,))
    bn_w_spec = pl.BlockSpec((BLOCK_N,), lambda m_idx, n_idx: (n_idx * BLOCK_N,))
    bn_b_spec = pl.BlockSpec((BLOCK_N,), lambda m_idx, n_idx: (n_idx * BLOCK_N,))
    out_spec = pl.BlockSpec((BLOCK_M, BLOCK_N), lambda m_idx, n_idx: (m_idx * BLOCK_M, n_idx * BLOCK_N))
    
    return pl.pallas_call(
        kernel_ref,
        out_shape=out_shape,
        grid=(grid_m, grid_n),
        in_specs=(x_spec, w_spec, b_spec, bn_w_spec, bn_b_spec),
        out_specs=out_spec,
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel")
        ),
    )(x, gemm_weight, gemm_bias, bn_weight, bn_bias)
