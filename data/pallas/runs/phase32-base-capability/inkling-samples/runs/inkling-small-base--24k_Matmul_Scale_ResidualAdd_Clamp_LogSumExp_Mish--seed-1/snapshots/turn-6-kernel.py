import jax
import jax.numpy as jnp
import jax.scipy.special
import jax.lax as lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

def workload(x, weight, bias):
    batch_size = x.shape[0]
    hidden_size = x.shape[1]
    batch_tile = 128
    
    def kernel(x_ref, w_ref, b_ref, out_ref):
        # Load block
        x_block = x_ref[...]
        # Compute matmul in float32
        x_f32 = x_block.astype(jnp.float32)
        w_f32 = w_ref[...].astype(jnp.float32)
        # x @ weight.T + bias
        # weight is [hidden, hidden], so weight.T is same
        mat = jnp.dot(x_f32, w_f32.T) + b_ref[...].astype(jnp.float32)
        
        # Scale by 2.0
        mat = mat * 2.0
        
        # Residual add: x + x
        mat = mat + mat
        
        # Clamp
        mat = jnp.clip(mat, -10.0, 10.0)
        
        # LogSumExp over axis=1, keepdims=True
        mat = jax.scipy.special.logsumexp(mat, axis=1, keepdims=True)
        
        # softplus = logaddexp(x, 0.0)
        softplus_x = jnp.logaddexp(mat, 0.0)
        
        # mish = x * tanh(softplus_x)
        mish_x = mat * jnp.tanh(softplus_x)
        
        # x = x * mish_x
        mat = mat * mish_x
        
        # Write out as bfloat16
        out_ref[...] = mat.astype(jnp.bfloat16)
    
    grid = (batch_size // batch_tile,)
    
    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((batch_size, 1), jnp.bfloat16),
        grid=grid,
        in_specs=(
            pl.BlockSpec((batch_tile, hidden_size), lambda i: (i * batch_tile, 0)),
            pl.no_block_spec,
            pl.no_block_spec,
        ),
        out_specs=pl.BlockSpec((batch_tile, 1), lambda i: (i * batch_tile, 0)),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel",)),
    )(x, weight, bias)
