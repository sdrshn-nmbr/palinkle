import jax
import jax.numpy as jnp
import pallas as pl
import jax.experimental.pallas as pl
import jax.experimental.pallas as pltpu

def matmul_scale_residual_clamp_logsumexp_mish_kernel(
    x_ref,
    weight_ref,
    bias_ref,
    out_ref,
):
    # Matmul: x @ weight + bias
    # x_ref shape: [M, K], weight_ref shape: [K, N], bias_ref shape: [N]
    # out_ref shape: [M, 1]
    
    m = x_ref.shape[0]
    k = x_ref.shape[1]
    n = weight_ref.shape[1]
    
    # Compute matmul in float32 for accumulation
    acc = jnp.zeros((m, n), dtype=jnp.float32)
    
    # Manual matmul loop
    for i in range(k):
        # x column i: [m, 1]
        # weight row i: [1, n]
        x_col = x_ref[:, i].astype(jnp.float32)  # [m]
        w_row = weight_ref[i, :].astype(jnp.float32)  # [n]
        # Outer product: [m, n]
        acc = acc + jnp.outer(x_col, w_row)
    
    # Add bias
    acc = acc + bias_ref.astype(jnp.float32)
    
    # Scale by 2.0
    x = acc * 2.0
    
    # ResidualAdd: x + x (which is just 2*x, but we do it explicitly)
    x = x + x
    
    # Clamp to [-10.0, 10.0]
    x = jnp.clip(x, -10.0, 10.0)
    
    # LogSumExp along axis 1 with keepdims
    x = jax.scipy.special.logsumexp(x, axis=1, keepdims=True)
    
    # Mish: x * tanh(logaddexp(x, 0.0))
    softplus_x = jnp.logaddexp(x, 0.0)
    mish_x = x * jnp.tanh(softplus_x)
    
    # Final output
    x = x * mish_x
    
    out_ref[...] = x.astype(x_ref.dtype)


def workload(x, weight, bias):
    block_size = 128
    
    return pl.pallas_call(
        matmul_scale_residual_clamp_logsumexp_mish_kernel,
        out_shape=jax.ShapeDtypeStruct((x.shape[0], 1), x.dtype),
        grid=(x.shape[0] // block_size,),
        in_specs=(
            pl.BlockSpec((block_size, block_size), lambda i: (i * block_size, 0)),
            pl.BlockSpec((block_size, block_size), lambda i: (0, i * block_size)),
            pl.BlockSpec((block_size,), lambda i: (i * block_size,)),
        ),
        out_specs=pl.BlockSpec((block_size, 1), lambda i: (i * block_size, 0)),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel",)
        ),
    )(x, weight, bias)
