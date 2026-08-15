import jax
import jax.numpy as jnp
import jax.pallas as pl
import jax.pallas as plp
import jax.interpreters.tpu as pltpu

def workload(x, weight, bias):
    """Matmul + GELU + Softmax kernel."""
    M, K = x.shape
    _, N = weight.shape
    
    # Block size for tiling - use multiples of 8 for bf16
    block_m = 128
    block_n = 128
    block_k = 128
    
    def kernel(ref_x, ref_weight, ref_bias, out_ref):
        # Compute matmul with bias
        # Accumulate in float32 for better precision
        def matmul_step(accum, _: pl.Slice):
            # Load blocks
            x_block = ref_x[pl.program_id(0) * block_m:(pl.program_id(0) + 1) * block_m, :]
            w_block = ref_weight[:, pl.program_id(1) * block_n:(pl.program_id(1) + 1) * block_n]
            # Compute partial matmul
            partial = jnp.dot(x_block, w_block).astype(jnp.float32)
            return accum + partial
        
        # Initialize accumulator
        accum = jnp.zeros((block_m, block_n), dtype=jnp.float32)
        
        # Tile over K dimension
        num_k_tiles = K // block_k
        for k_idx in range(num_k_tiles):
            accum = matmul_step(accum, None)
        
        # Add bias
        bias_block = ref_bias[pl.program_id(1) * block_n:(pl.program_id(1) + 1) * block_n]
        accum = accum + bias_block.astype(jnp.float32)
        
        # Apply GELU
        # GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        gelu_result = 0.5 * accum * (1 + jnp.tanh(
            jnp.sqrt(2.0 / jnp.pi) * (accum + 0.044715 * accum ** 3)
        ))
        
        # Apply softmax along axis 1 (last axis)
        # Subtract max for numerical stability
        max_val = jnp.max(gelu_result, axis=-1, keepdims=True)
        shifted = gelu_result - max_val
        exp_shifted = jnp.exp(shifted)
        sum_exp = jnp.sum(exp_shifted, axis=-1, keepdims=True)
        softmax_result = exp_shifted / sum_exp
        
        # Write output
        out_ref[...] = softmax_result.astype(jnp.bfloat16)
    
    # Grid dimensions
    grid_m = M // block_m
    grid_n = N // block_n
    
    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((M, N), jnp.bfloat16),
        grid=(grid_m, grid_n),
        in_specs=(
            pl.BlockSpec((block_m, K), lambda mi, ni: (mi * block_m, slice(None))),
            pl.BlockSpec((K, block_n), lambda mi, ni: (slice(None), ni * block_n)),
            pl.BlockSpec((block_n,), lambda mi, ni: (ni * block_n,)),
        ),
        out_specs=pl.BlockSpec((block_m, block_n), lambda mi, ni: (mi * block_m, ni * block_n)),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel")
        ),
    )(x, weight, bias)
