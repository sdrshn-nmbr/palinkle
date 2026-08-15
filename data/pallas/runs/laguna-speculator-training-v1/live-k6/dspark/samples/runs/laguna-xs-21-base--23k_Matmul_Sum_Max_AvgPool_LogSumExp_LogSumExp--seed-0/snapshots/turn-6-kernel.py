import jax
import jax.numpy as jnp
import jax.pallas as pl
import jax.pallas.lib as pllib
import jax.pallas.tpu as pltpu


def kernel(x_ref, weight_ref, bias_ref, out_ref):
    # Get program IDs for grid traversal
    m = pl.program_id(0)  # batch dimension
    
    # Compute matmul: x[m, :] @ weight[:, :] -> result[m, :]
    # x_ref has shape (8192,) for this row
    # weight_ref has shape (8192, 8192)
    
    # Perform matmul with accumulation in float32
    result = jnp.zeros((8192,), dtype=jnp.float32)
    for k in range(8192):
        result = result + x_ref[k] * weight_ref[k, :]
    
    # Add bias
    result = result + bias_ref[:]
    
    # Convert back to bfloat16 for subsequent operations
    result = result.astype(jnp.bfloat16)
    
    # Sum along axis 1 (which is now the feature dimension)
    # Since we're processing one row at a time, we sum all elements
    sum_result = jnp.sum(result, axis=0, keepdims=True)
    
    # Max
    max_result = jnp.max(sum_result, axis=0, keepdims=True)
    
    # Mean
    mean_result = jnp.mean(max_result, axis=0, keepdims=True)
    
    # LogSumExp twice
    lse1 = jax.scipy.special.logsumexp(mean_result, axis=0, keepdims=True)
    lse2 = jax.scipy.special.logsumexp(lse1, axis=0, keepdims=True)
    
    # Write output
    out_ref[...] = lse2


def workload(x, weight, bias):
    block_size = 128
    
    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((4096, 1), jnp.bfloat16),
        grid=(4096,),
        in_specs=(
            pl.BlockSpec((8192,), lambda m: (m,)),
            pl.BlockSpec((8192, 8192), lambda m: (m, slice(None))),
            pl.BlockSpec((8192,), lambda m: (slice(None),)),
        ),
        out_specs=pl.BlockSpec((1,), lambda m: (m,)),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel",)
        ),
    )(x, weight, bias)
