import jax
import jax.numpy as jnp
import jax.lax as lax
from jax import random
import jax.experimental.pallas as pl
import jax.experimental.pallas.tpu as pltpu

def mish(x):
    return x * jnp.tanh(jax.nn.softplus(x))

def workload(x, weight, bias):
    batch_size, in_features = x.shape
    out_features = weight.shape[1]
    
    def kernel(x_ref, w_ref, b_ref, out_ref):
        # Load full x row block and weight column block
        x_block = x_ref[...]
        w_block = w_ref[...]
        b_block = b_ref[...]
        
        # Matmul
        y = jnp.dot(x_block, w_block)
        
        # Add bias
        y = y + b_block
        
        # Mish twice
        y = mish(y)
        y = mish(y)
        
        out_ref[...] = y
    
    # Block shapes
    block_b = 128
    block_o = 128
    
    grid = (batch_size // block_b, out_features // block_o)
    
    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        grid=grid,
        in_specs=(
            pl.BlockSpec((block_b, in_features), lambda i, j: (i * block_b, 0)),
            pl.BlockSpec((in_features, block_o), lambda i, j: (0, j * block_o)),
            pl.BlockSpec((block_o,), lambda i, j: (j * block_o,)),
        ),
        out_specs=pl.BlockSpec((block_b, block_o), lambda i, j: (i * block_b, j * block_o)),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel", "parallel")),
    )(x, weight, bias)
