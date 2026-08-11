import jax
import jax.numpy as jnp
import jax.experimental.pallas as pl
import jax.experimental.pallas.tpu as pltpu

def workload(x, weight, bias):
    block_m = 128
    block_n = 128
    
    def kernel(x_ref, w_ref, b_ref, out_ref):
        x = x_ref[...].astype(jnp.float32)
        w = w_ref[...].astype(jnp.float32)
        b = b_ref[...].astype(jnp.float32)
        out = jnp.dot(x, w)
        out = out + b
        out = out * jax.nn.sigmoid(out)
        out = out * 2.0
        out_ref[...] = out.astype(jnp.bfloat16)
    
    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        grid=(x.shape[0] // block_m, x.shape[1] // block_n),
        in_specs=(
            pl.BlockSpec((block_m, x.shape[1]), lambda i, j: (i * block_m, 0)),
            pl.BlockSpec((weight.shape[0], block_n), lambda i, j: (0, j * block_n)),
            pl.BlockSpec((block_n,), lambda i, j: (j * block_n,)),
        ),
        out_specs=pl.BlockSpec((block_m, block_n), lambda i, j: (i * block_m, j * block_n)),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel", "parallel")),
    )(x, weight, bias)
