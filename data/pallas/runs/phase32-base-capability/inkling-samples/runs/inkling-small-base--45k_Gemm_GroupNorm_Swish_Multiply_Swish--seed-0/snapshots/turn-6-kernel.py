import jax
import jax.numpy as jnp
import jax.experimental.pallas as pl
import jax.experimental.pallas.tpu as pltpu

def matmul_kernel(x_ref, w_ref, b_ref, out_ref):
    # x_ref: (128, 8192) bf16
    # w_ref: (8192, 128) bf16
    # b_ref: (128,) bf16
    # out_ref: (128, 128) bf16
    out_ref[...] = jnp.dot(x_ref[...], w_ref[...]) + b_ref[...]

def post_kernel(y_ref, gn_w_ref, gn_b_ref, mult_w_ref, out_ref):
    # y_ref: (128, 8192) bf16
    # gn_w_ref: (8192,) bf16
    # gn_b_ref: (8192,) bf16
    # mult_w_ref: (8192,) bf16
    # out_ref: (128, 8192) bf16
    batch_size = 128
    num_groups = 256
    group_size = 32
    y_grouped = y_ref[...].reshape(batch_size, num_groups, group_size)
    mean = jnp.mean(y_grouped, axis=-1, keepdims=True)
    var = jnp.var(y_grouped, axis=-1, keepdims=True)
    y_norm = (y_grouped - mean) / jnp.sqrt(var + 1e-5)
    y_norm = y_norm.reshape(batch_size, 8192)
    x = y_norm * gn_w_ref[...] + gn_b_ref[...]
    x = x * jax.nn.sigmoid(x)
    x = x * mult_w_ref[...]
    x = x * jax.nn.sigmoid(x)
    out_ref[...] = x

def workload(x, gemm_weight, gemm_bias, gn_weight, gn_bias, multiply_weight):
    # Step 1: Gemm
    y = pl.pallas_call(
        matmul_kernel,
        out_shape=jax.ShapeDtypeStruct((4096, 8192), jnp.bfloat16),
        grid=(32, 64),
        in_specs=(
            pl.BlockSpec((128, 8192), lambda i, j: (i, 0)),
            pl.BlockSpec((8192, 128), lambda i, j: (0, j)),
            pl.BlockSpec((128,), lambda i, j: (j,)),
        ),
        out_specs=pl.BlockSpec((128, 128), lambda i, j: (i, j)),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel", "parallel")),
    )(x, gemm_weight, gemm_bias)
    
    # Step 2-6: GroupNorm + Swish + Multiply + Swish
    return pl.pallas_call(
        post_kernel,
        out_shape=jax.ShapeDtypeStruct((4096, 8192), jnp.bfloat16),
        grid=(32,),
        in_specs=(
            pl.BlockSpec((128, 8192), lambda i: (i, 0)),
            pl.BlockSpec((8192,), lambda i: (0,)),
            pl.BlockSpec((8192,), lambda i: (0,)),
            pl.BlockSpec((8192,), lambda i: (0,)),
        ),
        out_specs=pl.BlockSpec((128, 8192), lambda i: (i, 0)),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel",)),
    )(y, gn_weight, gn_bias, multiply_weight)
