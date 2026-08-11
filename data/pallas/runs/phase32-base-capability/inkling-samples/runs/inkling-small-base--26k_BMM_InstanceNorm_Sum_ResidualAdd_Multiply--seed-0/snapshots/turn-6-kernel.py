import jax
import jax.numpy as jnp
import jax.lax as lax
import jax.experimental.pallas as pl
import jax.experimental.pallas.tpu as pltpu

def workload(x, y, bmm_weight, bmm_bias, in_weight, in_bias):
    block_batch = 128
    batch = x.shape[0]
    grid = (batch // block_batch,)
    
    def kernel(x_ref, y_ref, bmm_weight_ref, bmm_bias_ref, in_weight_ref, in_bias_ref, out_ref):
        # Load block of x and y
        x_block = x_ref[...]
        y_block = y_ref[...]
        
        # Load full weights/biases (no_block_spec)
        w = bmm_weight_ref[...]
        b = bmm_bias_ref[...]
        iw = in_weight_ref[...]
        ib = in_bias_ref[...]
        
        # BMM: x @ w.T + b
        x_val = jnp.dot(x_block, w.T) + b
        
        # Instance norm: expand dims 2,3
        x_val = jnp.expand_dims(x_val, 2)
        x_val = jnp.expand_dims(x_val, 3)
        
        mean = jnp.mean(x_val, axis=(2, 3), keepdims=True)
        var = jnp.var(x_val, axis=(2, 3), keepdims=True)
        eps = 1e-05
        x_val = (x_val - mean) / jnp.sqrt(var + eps)
        
        # Apply instance weight/bias with reshape
        x_val = x_val * jnp.reshape(iw, (1, -1, 1, 1)) + jnp.reshape(ib, (1, -1, 1, 1))
        
        # Squeeze axes 3 then 2
        x_val = jnp.squeeze(x_val, axis=3)
        x_val = jnp.squeeze(x_val, axis=2)
        
        # Residual add and multiply
        x_val = x_val + y_block
        x_val = x_val * y_block
        
        out_ref[...] = x_val
    
    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        grid=grid,
        in_specs=(
            pl.BlockSpec((block_batch, x.shape[1]), lambda i: (i * block_batch, 0)),
            pl.BlockSpec((block_batch, y.shape[1]), lambda i: (i * block_batch, 0)),
            pl.no_block_spec,
            pl.no_block_spec,
            pl.no_block_spec,
            pl.no_block_spec,
        ),
        out_specs=pl.BlockSpec((block_batch, x.shape[1]), lambda i: (i * block_batch, 0)),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel",)),
    )(x, y, bmm_weight, bmm_bias, in_weight, in_bias)
