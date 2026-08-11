import jax
import jax.numpy as jnp
import jax.lax as lax
from jax import random
import jax.experimental.pallas as pl
import jax.experimental.pallas.tpu as pltpu

def workload(x, weight, bias, gn_weight, gn_bias):
    batch_size = 8192
    in_features = 4096
    out_features = 4096
    num_groups = 64
    group_size = out_features // num_groups  # 64
    
    batch_block = 128
    out_block = 128
    
    def kernel(x_ref, w_ref, b_ref, gn_w_ref, gn_b_ref, out_ref):
        # Load tiles
        x_tile = x_ref[...]  # (batch_block, in_features)
        w_tile = w_ref[...]  # (in_features, out_block)
        
        # Matmul in float32 for accumulation
        x_f32 = x_tile.astype(jnp.float32)
        w_f32 = w_tile.astype(jnp.float32)
        y = jnp.dot(x_f32, w_f32).astype(jnp.bfloat16)  # (batch_block, out_block)
        
        # Swish: x * sigmoid(x)
        y = y * jax.nn.sigmoid(y)
        
        # Add bias
        b_tile = b_ref[...]
        y = y + b_tile
        
        # GroupNorm over groups along feature axis
        # Tile covers out_block features = out_block // group_size groups
        groups_in_tile = out_block // group_size  # 2
        y_reshaped = y.reshape(batch_block, groups_in_tile, group_size)
        mean = jnp.mean(y_reshaped, axis=-1, keepdims=True)
        var = jnp.var(y_reshaped, axis=-1, keepdims=True)
        y_norm = (y_reshaped - mean) / jnp.sqrt(var + 1e-5)
        y = y_norm.reshape(batch_block, out_block)
        
        # Apply gn_weight and gn_bias
        gn_w_tile = gn_w_ref[...]
        gn_b_tile = gn_b_ref[...]
        y = y * gn_w_tile + gn_b_tile
        
        out_ref[...] = y
    
    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        grid=(batch_size // batch_block, out_features // out_block),
        in_specs=(
            pl.BlockSpec((batch_block, in_features), lambda i, j: (i * batch_block, 0)),
            pl.BlockSpec((in_features, out_block), lambda i, j: (0, j * out_block)),
            pl.BlockSpec((out_block,), lambda i, j: (j * out_block,)),
            pl.BlockSpec((out_block,), lambda i, j: (j * out_block,)),
            pl.BlockSpec((out_block,), lambda i, j: (j * out_block,)),
        ),
        out_specs=pl.BlockSpec((batch_block, out_block), lambda i, j: (i * batch_block, j * out_block)),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel", "parallel")),
    )(x, weight, bias, gn_weight, gn_bias)
