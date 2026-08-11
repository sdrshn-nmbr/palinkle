import jax
import jax.numpy as jnp
import jax.experimental.pallas as pl
import jax.experimental.pallas.tpu as pltpu

def workload(x, y, bmm_weight, bmm_bias, in_weight, in_bias):
    block = 128
    batch = x.shape[0]
    
    def kernel(x_ref, y_ref, w_ref, b_ref, iw_ref, ib_ref, out_ref):
        x_local = x_ref[...].astype(jnp.float32)
        y_local = y_ref[...].astype(jnp.float32)
        
        res = jnp.zeros((x_local.shape[0], w_ref.shape[1]), dtype=jnp.float32)
        chunk = 128
        # Use static loop with unroll? Just loop
        for i in range(0, w_ref.shape[0], chunk):
            w_slice = w_ref[i:i+chunk, :].astype(jnp.float32)
            x_slice = x_local[:, i:i+chunk]
            res += jnp.dot(x_slice, w_slice.T)
        res += b_ref[...].astype(jnp.float32)
        
        x_exp = jnp.expand_dims(res, 2)
        x_exp = jnp.expand_dims(x_exp, 3)
        mean = jnp.mean(x_exp, axis=(2, 3), keepdims=True)
        var = jnp.var(x_exp, axis=(2, 3), keepdims=True)
        eps = jnp.array(1e-05, dtype=jnp.float32)
        x_norm = (x_exp - mean) / jnp.sqrt(var + eps)
        
        iw = jnp.reshape(iw_ref[...].astype(jnp.float32), (1, -1, 1, 1))
        ib = jnp.reshape(ib_ref[...].astype(jnp.float32), (1, -1, 1, 1))
        x_norm = x_norm * iw + ib
        
        x_squeezed = jnp.squeeze(x_norm, axis=3)
        x_squeezed = jnp.squeeze(x_squeezed, axis=2)
        
        x_squeezed = x_squeezed + y_local
        x_squeezed = x_squeezed * y_local
        out_ref[...] = x_squeezed.astype(jnp.bfloat16)
    
    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        grid=(batch // block,),
        in_specs=(
            pl.BlockSpec((block, x.shape[1]), lambda i: (i, 0)),
            pl.BlockSpec((block, y.shape[1]), lambda i: (i, 0)),
            pl.BlockSpec((bmm_weight.shape[0], bmm_weight.shape[1]), lambda i: (0, 0)),
            pl.BlockSpec((bmm_bias.shape[0],), lambda i: (0,)),
            pl.BlockSpec((in_weight.shape[0],), lambda i: (0,)),
            pl.BlockSpec((in_bias.shape[0],), lambda i: (0,)),
        ),
        out_specs=pl.BlockSpec((block, x.shape[1]), lambda i: (i, 0)),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel",)),
    )(x, y, bmm_weight, bmm_bias, in_weight, in_bias)
