import jax
import jax.numpy as jnp
import jax.pallas as pl
import jax.pallas as plp
import jax.interpreters.tpu as pltpu


def matmul_kernel(x_ref, weight_ref, bias_ref, out_ref):
    """Pallas kernel for Matmul + Scaling + ResidualAdd."""
    m = x_ref.shape[0]
    k = x_ref.shape[1]
    n = weight_ref.shape[1]
    
    # Compute matmul + bias
    # Accumulate in float32 for better precision
    acc = jnp.zeros((n,), dtype=jnp.float32)
    for i in range(k):
        acc = acc + x_ref[i].astype(jnp.float32) * weight_ref[:, i].astype(jnp.float32)
    
    # Add bias
    acc = acc + bias_ref.astype(jnp.float32)
    
    # Scale by 0.5
    scaled = acc * 0.5
    
    # Residual add: scaled + original (matmul + bias result)
    result = scaled + acc
    
    out_ref[...] = result.astype(x_ref.dtype)


def workload(x, weight, bias):
    """Matmul + Scaling + ResidualAdd workload."""
    block_size = 128  # Block size for tiling
    
    # Grid: one block per output row
    grid = (x.shape[0] // block_size,)
    
    return pl.pallas_call(
        matmul_kernel,
        out_shape=jax.ShapeDtypeStruct((x.shape[0], weight.shape[1]), x.dtype),
        grid=grid,
        in_specs=(
            pl.BlockSpec((block_size, x.shape[1]), lambda i: (i * block_size, 0)),
            pl.BlockSpec((x.shape[1], block_size), lambda i: (0, i * block_size)),
            pl.BlockSpec((weight.shape[1],), lambda i: (0,)),
        ),
        out_specs=pl.BlockSpec((block_size, weight.shape[1]), lambda i: (i * block_size, 0)),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel",)
        ),
    )(x, weight, bias)
