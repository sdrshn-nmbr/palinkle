import jax
import jax.numpy as jnp
import jax.experimental.pallas as pl
import jax.experimental.pallas.tpu as pltpu

def workload(x, y, bmm_weight, bmm_bias, in_weight, in_bias):
    batch_size, in_features = x.shape
    out_features = bmm_bias.shape[0]
    
    batch_block = 128
    out_block = 128
    
    def kernel(x_ref, y_ref, w_ref, b_ref, in_w_ref, in_b_ref, out_ref):
        x_block = x_ref[...].astype(jnp.float32)
        w_block = w_ref[...].astype(jnp.float32)
        y_block = y_ref[...].astype(jnp.float32)
        b_block = b_ref[...].astype(jnp.float32)
        in_w_block = in_w_ref[...].astype(jnp.float32)
        in_b_block = in_b_ref[...].astype(jnp.float32)
        
        # BMM: x @ w.T + b
        out_block_val = jnp.dot(x_block, w_block.T) + b_block
        
        # Instance norm over axes 2,3 after expand_dims
        out_exp = jnp.expand_dims(out_block_val, 2)
        out_exp = jnp.expand_dims(out_exp, 3)
        
        mean = jnp.mean(out_exp, axis=(2, 3), keepdims=True)
        var = jnp.var(out_exp, axis=(2, 3), keepdims=True)
        eps = 1e-05
        out_norm = (out_exp - mean) / jnp.sqrt(var + eps)
        
        in_w_reshaped = jnp.reshape(in_w_block, (1, -1, 1, 1))
        in_b_reshaped = jnp.reshape(in_b_block, (1, -1, 1, 1))
        out_scaled = out_norm * in_w_reshaped + in_b_reshaped
        
        out_squeezed = jnp.squeeze(out_scaled, axis=3)
        out_squeezed = jnp.squeeze(out_squeezed, axis=2)
        
        out_final = out_squeezed + y_block
        out_final = out_final * y_block
        
        out_ref[...] = out_final.astype(jnp.bfloat16)
    
    grid = (batch_size // batch_block, out_features // out_block)
    
    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        grid=grid,
        in_specs=(
            pl.BlockSpec((batch_block, in_features), lambda i, j: (i * batch_block, 0)),
            pl.BlockSpec((batch_block, out_block), lambda i, j: (i * batch_block, j * out_block)),
            pl.BlockSpec((in_features, out_block), lambda i, j: (0, j * out_block)),
            pl.BlockSpec((out_block,), lambda i, j: (j * out_block,)),
            pl.BlockSpec((out_block,), lambda i, j: (j * out_block,)),
            pl.BlockSpec((out_block,), lambda i, j: (j * out_block,)),
        ),
        out_specs=pl.BlockSpec((batch_block, out_block), lambda i, j: (i * batch_block, j * out_block)),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel", "parallel")),
    )(x, y, bmm_weight, bmm_bias, in_weight, in_bias)
