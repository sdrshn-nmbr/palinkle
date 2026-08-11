import jax
import jax.numpy as jnp
import jax.experimental.pallas as pl
import jax.experimental.pallas.tpu as pltpu

def mish_kernel(x_ref, w_ref, b_ref, out_ref):
    # Load tiles and compute in float32
    x_tile = x_ref[...].astype(jnp.float32)
    w_tile = w_ref[...].astype(jnp.float32)
    b_tile = b_ref[...].astype(jnp.float32)
    
    # Matmul
    acc = jnp.dot(x_tile, w_tile)
    # Add bias (broadcast over M)
    acc = acc + b_tile
    
    # Mish 1: x * tanh(softplus(x))
    mish1 = acc * jnp.tanh(jax.nn.softplus(acc))
    # Mish 2
    mish2 = mish1 * jnp.tanh(jax.nn.softplus(mish1))
    
    out_ref[...] = mish2.astype(jnp.bfloat16)

def workload(x, weight, bias):
    M, K = x.shape
    K2, N = weight.shape
    assert K == K2
    
    block_m = 128
    block_n = 128
    
    grid = (M // block_m, N // block_n)
    
    return pl.pallas_call(
        mish_kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        grid=grid,
        in_specs=(
            pl.BlockSpec((block_m, K), lambda i, j: (i * block_m, 0)),
            pl.BlockSpec((K, block_n), lambda i, j: (0, j * block_n)),
            pl.BlockSpec((block_n,), lambda i, j: (j * block_n,)),
        ),
        out_specs=pl.BlockSpec((block_m, block_n), lambda i, j: (i * block_m, j * block_n)),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel", "parallel")),
    )(x, weight, bias)
