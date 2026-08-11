import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

def workload(x, weight, bias):
    M, K = x.shape
    K2, N = weight.shape
    assert K == K2
    
    block_m = 128
    block_n = 128
    
    def kernel(x_ref, w_ref, b_ref, out_ref):
        acc = jnp.zeros((block_m, block_n), dtype=jnp.float32)
        
        for k in range(0, K, 128):
            x_tile = jnp.astype(x_ref[:, k:k+128], jnp.float32)
            w_tile = jnp.astype(w_ref[k:k+128, :], jnp.float32)
            acc += jnp.dot(x_tile, w_tile)
        
        out = acc + jnp.astype(b_ref[:], jnp.float32)
        out = out * jax.nn.sigmoid(out)
        out = out * 2.0
        
        out_ref[...] = jnp.astype(out, jnp.bfloat16)
    
    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        grid=(M // block_m, N // block_n),
        in_specs=(
            pl.BlockSpec((block_m, K), lambda m, n: (m * block_m, 0)),
            pl.BlockSpec((K, block_n), lambda m, n: (0, n * block_n)),
            pl.BlockSpec((block_n,), lambda m, n: (n * block_n,)),
        ),
        out_specs=pl.BlockSpec((block_m, block_n), lambda m, n: (m * block_m, n * block_n)),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel", "parallel")),
    )(x, weight, bias)
