import jax
import jax.numpy as jnp
import jax.experimental.pallas as pl
import jax.experimental.pallas.tpu as pltpu

def workload(x, weight, bias):
    batch = x.shape[0]
    out_features = x.shape[1]
    in_features = weight.shape[0]
    block_b = 128
    block_o = 128

    def kernel(x_ref, w_ref, b_ref, out_ref):
        x_f = x_ref[...].astype(jnp.float32)
        w_f = w_ref[...].astype(jnp.float32)
        b_f = b_ref[...].astype(jnp.float32)

        out_f = jnp.dot(x_f, w_f)
        out_f = out_f + b_f[jnp.newaxis, :]
        out_f = out_f * 0.5
        out_f = jnp.clip(out_f, -2.0, 2.0)

        sqrt_2_over_pi = jnp.sqrt(2.0 / jnp.pi)
        out_f = 0.5 * out_f * (1.0 + jnp.tanh(sqrt_2_over_pi * (out_f + 0.044715 * (out_f ** 3))))

        out_ref[...] = out_f.astype(jnp.bfloat16)

    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        grid=(batch // block_b, out_features // block_o),
        in_specs=(
            pl.BlockSpec((block_b, in_features), lambda i, j: (i * block_b, 0)),
            pl.BlockSpec((in_features, block_o), lambda i, j: (0, j * block_o)),
            pl.BlockSpec((block_o,), lambda i, j: (j * block_o,)),
        ),
        out_specs=pl.BlockSpec((block_b, block_o), lambda i, j: (i * block_b, j * block_o)),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel", "parallel")),
    )(x, weight, bias)
