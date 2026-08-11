import jax
import jax.numpy as jnp
from jax import nn
from jax.experimental import pallas as pl
import jax.experimental.pallas.tpu as pltpu

def workload(x, gemm_weight, gemm_bias, gn_weight, gn_bias, multiply_weight):
    batch_size = 4096
    out_features = 8192
    num_groups = 256
    group_size = out_features // num_groups
    block_batch = 64

    def kernel(x_ref, w_ref, b_ref, gn_w_ref, gn_b_ref, mult_w_ref, out_ref):
        # Read block
        x_block = x_ref[...]
        # Matmul with float32 accumulation
        x_f32 = x_block.astype(jnp.float32)
        w_f32 = w_ref[...].astype(jnp.float32)
        y = jnp.dot(x_f32, w_f32.T) + b_ref[...].astype(jnp.float32)
        y = y.astype(x_block.dtype)

        # GroupNorm
        bs = y.shape[0]
        y_grouped = y.reshape(bs, num_groups, group_size)
        mean = jnp.mean(y_grouped, axis=-1, keepdims=True)
        var = jnp.var(y_grouped, axis=-1, keepdims=True)
        y_norm = (y_grouped - mean) / jnp.sqrt(var + 1e-5)
        y_norm = y_norm.reshape(bs, out_features)

        # Scale/shift
        y_norm = y_norm * gn_w_ref[...] + gn_b_ref[...]

        # Swish
        y_swish = y_norm * nn.sigmoid(y_norm)

        # Multiply
        y_mult = y_swish * mult_w_ref[...]

        # Swish again
        out = y_mult * nn.sigmoid(y_mult)

        out_ref[...] = out.astype(x_ref.dtype)

    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        grid=(batch_size // block_batch,),
        in_specs=(
            pl.BlockSpec((block_batch, out_features), lambda i: (i * block_batch, 0)),
            pl.no_block_spec,
            pl.no_block_spec,
            pl.no_block_spec,
            pl.no_block_spec,
            pl.no_block_spec,
        ),
        out_specs=pl.BlockSpec((block_batch, out_features), lambda i: (i * block_batch, 0)),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel",)),
    )(x, gemm_weight, gemm_bias, gn_weight, gn_bias, multiply_weight)
