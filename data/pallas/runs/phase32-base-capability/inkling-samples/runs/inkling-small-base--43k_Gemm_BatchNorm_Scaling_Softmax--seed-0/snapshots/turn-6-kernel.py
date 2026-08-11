import jax
import jax.numpy as jnp
import jax.lax as lax
from jax import random
import jax.experimental.pallas as pl
import jax.experimental.pallas.tpu as pltpu

def workload(x, weight, bias, bn_scale, bn_bias, bn_mean, bn_var, scale):
    # Tiled kernel over batch dimension (4096 -> 32 tiles of 128)
    batch_size = x.shape[0]
    out_features = x.shape[1]
    in_features = weight.shape[0]
    
    tile_batch = 128
    grid = (batch_size // tile_batch,)
    
    def kernel(x_ref, w_ref, b_ref, bs_ref, bb_ref, bm_ref, bv_ref, sc_ref, out_ref):
        # Load batch tile
        x_tile = x_ref[:, :]  # (128, 8192)
        
        # Initialize float32 accumulator
        accum = jnp.zeros((tile_batch, out_features), dtype=jnp.float32)
        
        # Tiled matmul over reduction dimension (8192) in chunks of 256
        chunk = 256
        for k in range(0, in_features, chunk):
            x_chunk = x_tile[:, k:k+chunk]  # (128, chunk)
            w_chunk = w_ref[k:k+chunk, :]     # (chunk, 8192)
            # jnp.dot: (128, chunk) @ (chunk, 8192) -> (128, 8192)
            accum += jnp.dot(x_chunk.astype(jnp.float32), w_chunk.astype(jnp.float32))
        
        # Add bias
        accum += b_ref[:].astype(jnp.float32)
        
        # BatchNorm: (x - mean) / sqrt(var + eps)
        bn_eps = 1e-05
        mean = bm_ref[:].astype(jnp.float32)
        var = bv_ref[:].astype(jnp.float32)
        normalized = (accum - mean) / jnp.sqrt(var + bn_eps)
        
        # Scale and shift
        bs = bs_ref[:].astype(jnp.float32)
        bb = bb_ref[:].astype(jnp.float32)
        accum = bs * normalized + bb
        
        # Scalar scale
        sc = sc_ref[0].astype(jnp.float32)
        accum = sc * accum
        
        # Softmax over axis=1 (feature dimension)
        # For numerical stability
        max_val = jnp.max(accum, axis=1, keepdims=True)
        exp_shifted = jnp.exp(accum - max_val)
        sum_exp = jnp.sum(exp_shifted, axis=1, keepdims=True)
        accum = exp_shifted / sum_exp
        
        out_ref[:, :] = accum.astype(jnp.bfloat16)
    
    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        grid=grid,
        in_specs=(
            pl.BlockSpec((tile_batch, out_features), lambda i: (i * tile_batch, 0)),
            pl.BlockSpec((in_features, out_features), lambda i: (0, 0)),
            pl.BlockSpec((out_features,), lambda i: (0,)),
            pl.BlockSpec((out_features,), lambda i: (0,)),
            pl.BlockSpec((out_features,), lambda i: (0,)),
            pl.BlockSpec((out_features,), lambda i: (0,)),
            pl.BlockSpec((out_features,), lambda i: (0,)),
            pl.BlockSpec((1,), lambda i: (0,)),
        ),
        out_specs=pl.BlockSpec((tile_batch, out_features), lambda i: (i * tile_batch, 0)),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel",)),
    )(x, weight, bias, bn_scale, bn_bias, bn_mean, bn_var, scale)
