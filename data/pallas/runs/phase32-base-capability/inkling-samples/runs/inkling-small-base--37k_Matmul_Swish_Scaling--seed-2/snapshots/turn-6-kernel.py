import jax
import jax.numpy as jnp
import jax.lax as lax
import jax.experimental.pallas as pl
import jax.experimental.pallas.tpu as pltpu

BLOCK_M = 128
BLOCK_N = 128
BLOCK_K = 128
IN_FEATURES = 8192

def workload(x, weight, bias):
    def kernel(x_ref, w_ref, b_ref, out_ref):
        pid_m = pl.program_id(0)
        pid_n = pl.program_id(1)
        m_start = pid_m * BLOCK_M
        n_start = pid_n * BLOCK_N
        
        acc = jnp.zeros((BLOCK_M, BLOCK_N), dtype=jnp.float32)
        
        # Static loop over reduction dimension
        for k in range(0, IN_FEATURES, BLOCK_K):
            x_slice = x_ref[m_start:m_start + BLOCK_M, k:k + BLOCK_K]
            w_slice = w_ref[k:k + BLOCK_K, n_start:n_start + BLOCK_N]
            acc += jnp.dot(x_slice.astype(jnp.float32), w_slice.astype(jnp.float32))
        
        # Add bias
        b_slice = b_ref[n_start:n_start + BLOCK_N]
        result = acc + b_slice.astype(jnp.float32)
        
        # Swish: x * sigmoid(x)
        result = result * jax.nn.sigmoid(result)
        
        # Scale by 2.0
        result = result * 2.0
        
        out_ref[m_start:m_start + BLOCK_M, n_start:n_start + BLOCK_N] = result.astype(jnp.bfloat16)
    
    out_shape = jax.ShapeDtypeStruct(x.shape, x.dtype)
    
    return pl.pallas_call(
        kernel,
        out_shape=out_shape,
        grid=(x.shape[0] // BLOCK_M, x.shape[1] // BLOCK_N),
        in_specs=(
            pl.no_block_spec,
            pl.no_block_spec,
            pl.no_block_spec,
        ),
        out_specs=pl.no_block_spec,
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel", "parallel")),
    )(x, weight, bias)
