import jax
import jax.numpy as jnp
import pallas as pl
import pallas.lib as pllib
import triton.tools as pltpu

def kernel(x_ref, weight_ref, bias_ref, add_value_ref, out_ref):
    # Get program indices
    m = pl.program_id(0)  # batch dimension
    n = pl.program_id(1)  # output dimension
    
    # Block size for matmul
    block_m = 128
    block_n = 128
    block_k = 8
    
    # Accumulate in float32 for better precision
    acc = jnp.zeros((block_m, block_n), dtype=jnp.float32)
    
    # Perform matrix multiplication with tiling
    for k in range(0, x_ref.shape[1] // block_k):
        x_block = x_ref[m * block_m:(m + 1) * block_m, k * block_k:(k + 1) * block_k]
        w_block = weight_ref[k * block_k:(k + 1) * block_k, n * block_n:(n + 1) * block_n]
        acc = acc + jnp.dot(x_block, w_block)
    
    # Add bias
    acc = acc + bias_ref[n * block_n:(n + 1) * block_n]
    
    # Add add_value
    acc = acc + add_value_ref[n * block_n:(n + 1) * block_n]
    
    # Apply activations: swish -> tanh -> gelu -> clip
    acc = jax.nn.swish(acc)
    acc = jnp.tanh(acc)
    acc = jax.nn.gelu(acc)
    acc = jnp.clip(acc, -1.0, 1.0)
    
    # Write output
    out_ref[m * block_m:(m + 1) * block_m, n * block_n:(n + 1) * block_n] = acc

def workload(x, weight, bias, add_value):
    block_size = 128
    
    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((x.shape[0], x.shape[1]), x.dtype),
        grid=(x.shape[0] // block_size, x.shape[1] // block_size),
        in_specs=(
            pl.BlockSpec((block_size, block_size), lambda i: (0, i)),
            pl.BlockSpec((block_size, block_size), lambda i: (i, 0)),
            pl.BlockSpec((block_size,), lambda i: (i,)),
            pl.BlockSpec((block_size,), lambda i: (i,)),
        ),
        out_specs=pl.BlockSpec((block_size, block_size), lambda i: (i, 0)),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel")
        ),
    )(x, weight, bias, add_value)
