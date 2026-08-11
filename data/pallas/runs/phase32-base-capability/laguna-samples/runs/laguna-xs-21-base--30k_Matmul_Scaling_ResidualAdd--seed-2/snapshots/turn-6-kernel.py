import jax
import jax.numpy as jnp
import pallas as pl
import pallas.lib as pllib
import jax.pallas as pl

def matmul_kernel(x_ref, weight_ref, bias_ref, out_ref, program_id):
    """Kernel that computes: (x @ weight + bias) * 0.5 + (x @ weight + bias)"""
    m = x_ref.shape[0]
    k = x_ref.shape[1]
    n = weight_ref.shape[1]
    
    # Get the block index
    block_m = program_id(0)
    
    # Block size for tiling
    block_k = 128
    block_n = 128
    
    # Accumulator in float32 for better precision
    acc = jnp.zeros((block_m * 128, n), dtype=jnp.float32)
    
    # Compute matmul with tiling
    for i in range(0, k, block_k):
        for j in range(0, n, block_n):
            # Get blocks
            x_block = x_ref[block_m * 128:block_m * 128 + 128, i:i + block_k]
            w_block = weight_ref[i:i + block_k, j:j + block_n]
            
            # Compute partial product
            partial = jnp.dot(x_block, w_block, precision=jax.lax.Precision.HIGHEST)
            acc = acc.at[:, j:j + block_n].add(partial.astype(jnp.float32))
    
    # Add bias
    acc = acc + bias_ref[jnp.newaxis, :]
    
    # Store result
    out_ref[block_m * 128:block_m * 128 + 128, :] = acc.astype(x_ref.dtype)


def workload(x, weight, bias):
    """Compute: (x @ weight + bias) * 0.5 + (x @ weight + bias)"""
    m, k = x.shape
    k2, n = weight.shape
    
    # Grid: tile along batch dimension
    grid = (m // 128,)
    
    # Block spec for x: [128, 4096]
    x_spec = pl.BlockSpec((128, k), lambda i: (i[0] * 128, slice(None)))
    
    # Block spec for weight: [4096, 4096]
    weight_spec = pl.BlockSpec((k, n), lambda i: (slice(None), i[1] if len(i) > 1 else slice(None)))
    
    # Block spec for bias: scalar per output feature
    bias_spec = pl.BlockSpec((n,), lambda i: (slice(None),))
    
    # Block spec for output: [128, 4096]
    out_spec = pl.BlockSpec((128, n), lambda i: (i[0] * 128, slice(None)))
    
    return pl.pallas_call(
        matmul_kernel,
        out_shape=jax.ShapeDtypeStruct((m, n), x.dtype),
        grid=grid,
        in_specs=(x_spec, weight_spec, bias_spec),
        out_specs=(out_spec,),
        compiler_params=jax.xla.compiler.TpuCompilerParams(
            dimension_semantics=("parallel", "dot_operand")
        ),
    )(x, weight, bias)
