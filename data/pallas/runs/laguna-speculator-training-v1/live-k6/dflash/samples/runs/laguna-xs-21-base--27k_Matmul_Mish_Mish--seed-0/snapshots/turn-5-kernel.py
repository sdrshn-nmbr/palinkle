import jax
import jax.numpy as jnp
import jax.pallas as pl
import jax.pallas as plp
import jax.interpreters.tpu as pltpu


def matmul_mish_mish_kernel(x_ref, weight_ref, bias_ref, out_ref):
    """Pallas kernel for Matmul + Mish + Mish."""
    # Get grid indices
    m = pl.program_id(0)
    k = pl.program_id(1)
    
    # Block sizes
    block_m = 64
    block_k = 64
    block_n = 64
    
    # Accumulate in float32 for better precision
    acc = jnp.zeros(block_n, dtype=jnp.float32)
    
    # Matmul: x[m, :] @ weight[:, n] + bias[n]
    for kk in range(0, 8192, block_k):
        # Load x block [block_m, block_k]
        x_block = x_ref[m * block_m:(m + 1) * block_m, kk:kk + block_k]
        # Load weight block [block_k, block_n]
        weight_block = weight_ref[kk:kk + block_k, k * block_n:(k + 1) * block_n]
        
        # Compute partial matmul and accumulate
        acc = acc + jnp.dot(x_block.astype(jnp.float32), weight_block.astype(jnp.float32))
    
    # Add bias
    acc = acc + bias_ref[k * block_n:(k + 1) * block_n].astype(jnp.float32)
    
    # Convert to bfloat16 for Mish computation
    acc_bf16 = acc.astype(jnp.bfloat16)
    
    # First Mish: x * tanh(softplus(x))
    mish1 = acc_bf16 * jnp.tanh(jnp.nn.softplus(acc_bf16))
    
    # Second Mish: x * tanh(softplus(x))
    mish2 = mish1 * jnp.tanh(jnp.nn.softplus(mish1))
    
    # Write output
    out_ref[m * block_m:(m + 1) * block_m, k * block_n:(k + 1) * block_n] = mish2


def workload(x, weight, bias):
    """Compute Matmul + Mish + Mish."""
    block_m = 64
    block_k = 64
    block_n = 64
    
    # Grid dimensions
    num_blocks_m = x.shape[0] // block_m
    num_blocks_n = weight.shape[1] // block_n
    
    return pl.pallas_call(
        matmul_mish_mish_kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        grid=(num_blocks_m, num_blocks_n),
        in_specs=(
            pl.BlockSpec((x.shape[0], block_k), lambda m, n: (m * block_m, 0)),
            pl.BlockSpec((block_k, weight.shape[1]), lambda m, n: (0, n * block_n)),
            pl.BlockSpec((weight.shape[1],), lambda m, n: (n * block_n,)),
        ),
        out_specs=pl.BlockSpec((block_m, block_n), lambda m, n: (m * block_m, n * block_n)),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel")
        ),
    )(x, weight, bias)
