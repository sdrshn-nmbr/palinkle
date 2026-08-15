import jax
import jax.numpy as jnp
import pallas as pl
import pallas.lib as pllib
import pallas.tpu as pltpu


def kernel(x_ref, weight_ref, bias_ref, out_ref):
    # Matmul: x @ weight + bias
    # x_ref shape: [4096, 8192], weight_ref shape: [8192, 8192], bias_ref shape: [8192]
    # We compute row by row
    
    m, k = x_ref.shape
    _, n = weight_ref.shape
    
    # Get current row index
    row = pl.program_id(0)
    
    # Compute matmul for this row
    # Accumulate in float32 for better precision
    acc = jnp.zeros((n,), dtype=jnp.float32)
    
    # Tile along the k dimension
    tile_size = 128  # TPU-friendly tile size
    
    for tile_start in range(0, k, tile_size):
        tile_end = min(tile_start + tile_size, k)
        
        # Load x row tile [tile_size, 8192]
        x_tile = x_ref[row, tile_start:tile_end].astype(jnp.float32)
        # Load weight tile [8192, tile_size]
        weight_tile = weight_ref[tile_start:tile_end, :].astype(jnp.float32)
        
        # Compute partial matmul
        partial = jnp.dot(x_tile[0], weight_tile)  # [8192]
        acc = acc + partial
    
    # Add bias
    acc = acc + bias_ref[:].astype(jnp.float32)
    
    # Scale by 2.0
    x = acc * 2.0
    
    # ResidualAdd: x + x
    x = x + x
    
    # Clamp between -10.0 and 10.0
    x = jnp.clip(x, -10.0, 10.0)
    
    # LogSumExp along axis 0 (since we're working with a single row vector)
    # For a 1D array, logsumexp gives a scalar
    # We need to reshape to [1] for output
    x = jnp.expand_dims(jnp.logsumexp(x, axis=0, keepdims=True), 0)
    
    # Mish: x * tanh(logaddexp(x, 0.0))
    softplus_x = jnp.logaddexp(x, 0.0)
    mish_x = x * jnp.tanh(softplus_x)
    
    # Final multiply
    x = x * mish_x
    
    # Write output
    out_ref[:] = x.astype(x_ref.dtype)


def workload(x, weight, bias):
    block_size = 128
    
    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((4096, 1), x.dtype),
        grid=(4096,),
        in_specs=(
            pl.BlockSpec((1, 8192), lambda i: (i, 0)),  # x: one row at a time
            pl.BlockSpec((8192, 8192), lambda i: (0, 0)),  # weight: full matrix
            pl.BlockSpec((8192,), lambda i: (0,)),  # bias: full vector
        ),
        out_specs=pl.BlockSpec((1, 1), lambda i: (i, 0)),  # output: one element per row
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel",)
        ),
    )(x, weight, bias)
