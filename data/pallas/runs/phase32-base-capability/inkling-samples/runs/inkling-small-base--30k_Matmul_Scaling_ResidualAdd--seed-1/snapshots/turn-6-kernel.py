import jax
import jax.numpy as jnp
import jax.experimental.pallas as pl
import jax.experimental.pallas.tpu as pltpu

def workload(x, weight, bias):
    batch_size, in_features = x.shape
    out_features = weight.shape[1]
    
    def kernel(x_ref, w_ref, b_ref, out_ref):
        # Load tiles
        x_tile = x_ref[...]  # (128, 4096) bf16
        w_tile = w_ref[...]  # (4096, 128) bf16
        b_tile = b_ref[...]  # (128,) bf16
        
        # Compute matmul in float32
        accum = jnp.dot(x_tile.astype(jnp.float32), w_tile.astype(jnp.float32))
        # Add bias
        accum = accum + b_tile.astype(jnp.float32)[None, :]
        
        # Preserve semantics: original = accum; accum = accum * 0.5; accum = accum + original
        original = accum
        accum = accum * 0.5
        accum = accum + original
        
        out_ref[...] = accum.astype(jnp.bfloat16)
    
    block_batch = 128
    block_out = 128
    
    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        grid=(batch_size // block_batch, out_features // block_out),
        in_specs=(
            pl.BlockSpec((block_batch, in_features), lambda i, j: (i, 0)),
            pl.BlockSpec((in_features, block_out), lambda i, j: (0, j)),
            pl.BlockSpec((block_out,), lambda i, j: (j,)),
        ),
        out_specs=pl.BlockSpec((block_batch, block_out), lambda i, j: (i, j)),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel", "parallel")),
    )(x, weight, bias)
