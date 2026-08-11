import jax
import jax.numpy as jnp
import jax.experimental.pallas as pl
import jax.experimental.pallas.tpu as pltpu

def workload(x, weight, bias):
    def kernel(x_ref, w_ref, b_ref, out_ref):
        # Load blocks and cast to float32 for accumulation
        x = x_ref[...].astype(jnp.float32)
        w = w_ref[...].astype(jnp.float32)
        b = b_ref[...].astype(jnp.float32)
        # Matmul
        acc = jnp.dot(x, w)
        # Add bias (broadcast over batch dimension)
        acc = acc + b
        # Multiply by 2.0
        acc = acc * 2.0
        # Leaky ReLU: where >= 0 keep, else * 0.1
        acc = jnp.where(acc >= 0, acc, acc * 0.1)
        out_ref[...] = acc.astype(jnp.bfloat16)

    batch_size, in_features = x.shape
    out_features = weight.shape[1]
    block_b = 128
    block_o = 128
    grid = (batch_size // block_b, out_features // block_o)

    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        grid=grid,
        in_specs=(
            pl.BlockSpec((block_b, in_features), lambda i, j: (i, 0)),
            pl.BlockSpec((in_features, block_o), lambda i, j: (0, j)),
            pl.BlockSpec((block_o,), lambda i, j: (j,)),
        ),
        out_specs=pl.BlockSpec((block_b, block_o), lambda i, j: (i, j)),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel", "parallel")),
    )(x, weight, bias)
