import jax
import jax.numpy as jnp
import jax.lax as lax
from jax import random
import jax.experimental.pallas as pl
import jax.experimental.pallas.tpu as pltpu

def workload(x, weight, bias, gn_weight, gn_bias):
    batch = 128
    out_features = 4096
    
    def kernel(x_ref, w_ref, b_ref, gn_w_ref, gn_b_ref, out_ref):
        # Load block
        x_block = x_ref[...].astype(jnp.float32)
        w = w_ref[...].astype(jnp.float32)
        
        # Matmul with float32 accumulation
        y = jnp.dot(x_block, w)
        
        # Swish: x * sigmoid(x)
        y = y * jax.nn.sigmoid(y)
        
        # Add bias
        b = b_ref[...].astype(jnp.float32)
        y = y + b
        
        # GroupNorm
        # Reshape to (-1, num_groups, group_size)
        num_groups = 64
        group_size = out_features // num_groups
        y_reshaped = y.reshape(-1, num_groups, group_size)
        
        mean = jnp.mean(y_reshaped, axis=-1, keepdims=True)
        var = jnp.var(y_reshaped, axis=-1, keepdims=True)
        y_norm = (y_reshaped - mean) / jnp.sqrt(var + 1e-5)
        y = y_norm.reshape(-1, out_features)
        
        # Apply gn_weight and gn_bias
        gn_w = gn_w_ref[...].astype(jnp.float32)
        gn_b = gn_b_ref[...].astype(jnp.float32)
        y = y * gn_w + gn_b
        
        out_ref[...] = y.astype(jnp.bfloat16)
    
    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        grid=(x.shape[0] // batch,),
        in_specs=(
            pl.BlockSpec((batch, out_features), lambda i: (i * batch, 0)),
            pl.no_block_spec,
            pl.no_block_spec,
            pl.no_block_spec,
            pl.no_block_spec,
        ),
        out_specs=pl.BlockSpec((batch, out_features), lambda i: (i * batch, 0)),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel",)),
    )(x, weight, bias, gn_weight, gn_bias)
