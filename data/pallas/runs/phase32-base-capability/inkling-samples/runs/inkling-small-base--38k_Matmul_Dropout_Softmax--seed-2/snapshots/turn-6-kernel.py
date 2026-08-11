import jax
import jax.numpy as jnp
import jax.experimental.pallas as pl
import jax.experimental.pallas.tpu as pltpu

def workload(x, weight, bias):
    # Stage 1: matmul + bias
    def matmul_kernel(x_ref, w_ref, b_ref, out_ref):
        x_f = x_ref.astype(jnp.float32)
        w_f = w_ref.astype(jnp.float32)
        out_f = jnp.dot(x_f, w_f)
        b_f = b_ref.astype(jnp.float32)
        out_f = out_f + b_f[None, :]
        out_ref[...] = out_f.astype(jnp.bfloat16)

    # Stage 2: softmax over axis=1
    def softmax_kernel(x_ref, out_ref):
        x_f = x_ref.astype(jnp.float32)
        max_f = jnp.max(x_f, axis=1, keepdims=True)
        exp_f = jnp.exp(x_f - max_f)
        sum_f = jnp.sum(exp_f, axis=1, keepdims=True)
        out_f = exp_f / sum_f
        out_ref[...] = out_f.astype(jnp.bfloat16)

    # Matmul grid: (32, 64) for (4096/128, 8192/128)
    out_shape = jax.ShapeDtypeStruct(x.shape, x.dtype)
    matmul_out = pl.pallas_call(
        matmul_kernel,
        out_shape=out_shape,
        grid=(32, 64),
        in_specs=(
            pl.BlockSpec((128, 8192), lambda i, j: (i * 128, 0)),
            pl.BlockSpec((8192, 128), lambda i, j: (0, j * 128)),
            pl.BlockSpec((128,), lambda i, j: (j * 128,)),
        ),
        out_specs=pl.BlockSpec((128, 128), lambda i, j: (i * 128, j * 128)),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel", "parallel")),
    )(x, weight, bias)

    # Softmax grid: (32,) for (4096/128,)
    softmax_out = pl.pallas_call(
        softmax_kernel,
        out_shape=out_shape,
        grid=(32,),
        in_specs=(
            pl.BlockSpec((128, 8192), lambda i: (i * 128, 0)),
        ),
        out_specs=pl.BlockSpec((128, 8192), lambda i: (i * 128, 0)),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel",)),
    )(matmul_out)

    return softmax_out
