import jax
import jax.numpy as jnp
import jax.pallas as pl
import jax.pallas as plp
import jax.interpreters.tpu as pltpu


def matmul_dropout_softmax_kernel(
    x_ref,
    weight_ref,
    bias_ref,
    out_ref,
):
    # Get program IDs for tiling
    m_block = pl.program_id(0)  # batch dimension
    n_block = pl.program_id(1)  # output feature dimension
    
    # Block sizes
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 128
    
    # Accumulator in float32
    acc = jnp.zeros((BLOCK_M, BLOCK_N), dtype=jnp.float32)
    
    # Matmul: x @ weight.T
    # x: (BLOCK_M, K) bfloat16
    # weight.T: (K, BLOCK_N) bfloat16
    # We need to tile over K dimension
    
    # Get the shapes
    x_shape = x_ref.shape
    weight_shape = weight_ref.shape
    
    # Compute the number of K tiles
    k_tiles = (x_shape[1] + BLOCK_K - 1) // BLOCK_K
    
    for k_block in range(k_tiles):
        # Load x tile
        x_start_k = k_block * BLOCK_K
        x_end_k = min(x_start_k + BLOCK_K, x_shape[1])
        x_tile = x_ref[
            :, 
            x_start_k:x_end_k
        ]
        
        # Load weight tile (weight.T)
        # weight_ref is (8192, 8192), we need weight.T which is (8192, 8192)
        # For weight.T, we access weight_ref[:, x_start_k:x_end_k]
        w_tile = weight_ref[
            :, 
            x_start_k:x_end_k
        ]
        
        # Convert to float32 for accumulation
        x_f32 = x_tile.astype(jnp.float32)
        w_f32 = w_tile.astype(jnp.float32)
        
        # Accumulate
        acc = acc + jnp.dot(x_f32, w_f32.T)
    
    # Add bias
    # bias_ref is (8192,) - we need to broadcast to (BLOCK_M, BLOCK_N)
    bias_tile = bias_ref[:BLOCK_N]
    acc = acc + bias_tile[None, :]
    
    # Softmax along axis 1
    # Subtract max for numerical stability
    max_val = jnp.max(acc, axis=1, keepdims=True)
    shifted = acc - max_val
    exp_shifted = jnp.exp(shifted)
    sum_exp = jnp.sum(exp_shifted, axis=1, keepdims=True)
    softmax_result = exp_shifted / sum_exp
    
    # Convert back to bfloat16 and store
    out_ref[...] = softmax_result.astype(jnp.bfloat16)


def workload(x, weight, bias):
    batch_size = x.shape[0]
    out_features = x.shape[1]
    
    BLOCK_M = 128
    BLOCK_N = 128
    
    grid = (batch_size // BLOCK_M, out_features // BLOCK_N)
    
    return pl.pallas_call(
        matmul_dropout_softmax_kernel,
        out_shape=jax.ShapeDtypeStruct((batch_size, out_features), jnp.bfloat16),
        grid=grid,
        in_specs=(
            pl.BlockSpec((BLOCK_M, -1), lambda mi, ni: (mi * BLOCK_M, slice(None))),
            pl.BlockSpec((-1, BLOCK_N), lambda mi, ni: (slice(None), ni * BLOCK_N)),
            pl.BlockSpec((BLOCK_N,), lambda mi, ni: (ni * BLOCK_N,)),
        ),
        out_specs=pl.BlockSpec((BLOCK_M, BLOCK_N), lambda mi, ni: (mi * BLOCK_M, ni * BLOCK_N)),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel")
        ),
    )(x, weight, bias)
