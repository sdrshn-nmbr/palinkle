import jax
import jax.numpy as jnp
import jax.pallas as pl
import jax.pallas as plp
import jax.interpreters.tpu as pltpu

def workload(x, weight, bias):
    """Matmul + Mish + Mish kernel."""
    M, K = x.shape
    _, N = weight.shape
    
    # Block size for TPU - multiples of 8 for bf16
    block_m = 128
    block_n = 128
    block_k = 128
    
    def matmul_mish_mish_kernel(
        x_ref, weight_ref, bias_ref, out_ref
    ):
        # Get program indices
        m_block = pl.program_id(0)
        n_block = pl.program_id(1)
        
        # Compute tile bounds
        m_start = m_block * block_m
        n_start = n_block * block_n
        
        # Initialize accumulator in float32
        acc = jnp.zeros((block_m, block_n), dtype=jnp.float32)
        
        # Matmul with accumulation
        for k_block in range(K // block_k):
            k_start = k_block * block_k
            
            # Load tiles
            x_tile = x_ref[
                m_start:m_start + block_m,
                k_start:k_start + block_k
            ]
            w_tile = weight_ref[
                k_start:k_start + block_k,
                n_start:n_start + block_n
            ]
            
            # Compute partial matmul and accumulate
            acc = acc + jnp.dot(x_tile, w_tile)
        
        # Add bias (broadcast along m dimension)
        bias_tile = bias_ref[n_start:n_start + block_n]
        acc = acc + bias_tile[None, :]
        
        # Convert to bf16 for Mish computation
        acc_bf16 = acc.astype(jnp.bfloat16)
        
        # Mish activation: x * tanh(softplus(x))
        def mish(x):
            return x * jnp.tanh(jnp.nn.softplus(x))
        
        # Apply Mish twice
        result = mish(mish(acc_bf16))
        
        # Write output
        out_ref[
            m_start:m_start + block_m,
            n_start:n_start + block_n
        ] = result
    
    # Grid dimensions
    grid = (M // block_m, N // block_n)
    
    return pl.pallas_call(
        matmul_mish_mish_kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        grid=grid,
        in_specs=(
            pl.BlockSpec((block_m, block_k), lambda m, n, _: (m * block_m, 0)),
            pl.BlockSpec((block_k, block_n), lambda m, n, _: (0, n * block_n)),
            pl.BlockSpec((block_n,), lambda m, n, _: (n * block_n,)),
        ),
        out_specs=pl.BlockSpec((block_m, block_n), lambda m, n: (m * block_m, n * block_n)),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel")
        ),
    )(x, weight, bias)
