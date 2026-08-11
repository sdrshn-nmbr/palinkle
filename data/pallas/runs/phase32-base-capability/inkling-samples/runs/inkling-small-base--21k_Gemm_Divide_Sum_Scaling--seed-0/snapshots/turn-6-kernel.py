import jax
import jax.numpy as jnp
import jax.lax as lax
import jax.experimental.pallas as pl
import jax.experimental.pallas.tpu as pltpu

BATCH_BLOCK = 128
CONTRACT_BLOCK = 256

def workload(x, weight):
    def kernel(x_ref, weight_ref, out_ref):
        # Accumulate float32 partial sums over contraction dimension
        acc = jnp.zeros((BATCH_BLOCK, 8192), dtype=jnp.float32)
        for j in range(0, 8192, CONTRACT_BLOCK):
            x_chunk = x_ref[:, j:j+CONTRACT_BLOCK].astype(jnp.float32)
            w_chunk = weight_ref[:, j:j+CONTRACT_BLOCK].astype(jnp.float32)
            # w_chunk is (8192, CONTRACT_BLOCK); transpose to (CONTRACT_BLOCK, 8192)
            acc += jnp.dot(x_chunk, w_chunk.T)
        # Divide by 2.0
        acc = acc / 2.0
        # Sum over axis 1, keepdims=True
        acc = jnp.sum(acc, axis=1, keepdims=True)
        # Scale by 1.5
        acc = acc * 1.5
        out_ref[...] = acc.astype(jnp.bfloat16)

    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((x.shape[0], 1), jnp.bfloat16),
        grid=(x.shape[0] // BATCH_BLOCK,),
        in_specs=(
            pl.BlockSpec((BATCH_BLOCK, 8192), lambda i: (i * BATCH_BLOCK, 0)),
            pl.no_block_spec,
        ),
        out_specs=pl.BlockSpec((BATCH_BLOCK, 1), lambda i: (i * BATCH_BLOCK, 0)),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel",)),
    )(x, weight)
