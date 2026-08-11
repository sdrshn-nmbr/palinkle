import jax
import jax.numpy as jnp
import jax.experimental.pallas as pl
import jax.experimental.pallas.tpu as pltpu

def workload(x, weight, bias, gn_weight, gn_bias):
    batch_size = x.shape[0]
    out_features = weight.shape[1]
    block_batch = 128
    
    def kernel(x_ref, weight_ref, bias_ref, gn_weight_ref, gn_bias_ref, out_ref):
        # Load input tile
        x_tile = x_ref[...]
        # Load full weight (no block spec)
        w = weight_ref[...]
        # Matmul
        y = jnp.dot(x_tile, w)
        # Swish: x * sigmoid(x)
        y = y * jax.nn.sigmoid(y)
        # Add bias
        y = y + bias_ref[...]
        # GroupNorm
        batch = y.shape[0]
        num_groups = 64
        group_size = out_features // num_groups
        y_reshaped = y.reshape(batch, num_groups, group_size)
        mean = jnp.mean(y_reshaped, axis=-1, keepdims=True)
        var = jnp.var(y_reshaped, axis=-1, keepdims=True)
        y_norm = (y_reshaped - mean) / jnp.sqrt(var + 1e-5)
        y = y_norm.reshape(batch, out_features)
        # Apply gn weight and bias
        y = y * gn_weight_ref[...] + gn_bias_ref[...]
        out_ref[...] = y
    
    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        grid=(batch_size // block_batch,),
        in_specs=(
            pl.BlockSpec((block_batch, out_features), lambda i: (i * block_batch, 0)),
            pl.no_block_spec,
            pl.no_block_spec,
            pl.no_block_spec,
            pl.no_block_spec,
        ),
        out_specs=pl.BlockSpec((block_batch, out_features), lambda i: (i * block_batch, 0)),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel",)),
    )(x, weight, bias, gn_weight, gn_bias)
