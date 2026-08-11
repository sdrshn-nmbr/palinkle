import jax
import jax.numpy as jnp
import jax.scipy.special as jss
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

def workload(x, weight, bias):
    def kernel(x_ref, weight_ref, bias_ref, out_ref):
        i = pl.program_id(0)
        # Load x row slice (1, 8192) -> we index directly
        x_row = x_ref[i, :]
        # Accumulate matmul result in float32
        row = jnp.zeros((8192,), dtype=jnp.float32)
        block = 256
        for k in range(0, 8192, block):
            x_slice = x_row[k:k+block].astype(jnp.float32)
            w_slice = weight_ref[k:k+block, :].astype(jnp.float32)
            row += jnp.dot(x_slice, w_slice)
        # Add bias
        row += bias_ref[:].astype(jnp.float32)
        # Sequential reductions exactly as AST
        val = jnp.sum(row, axis=0, keepdims=True)
        val = jnp.max(val, axis=0, keepdims=True)
        val = jnp.mean(val, axis=0, keepdims=True)
        val = jss.logsumexp(val, axis=0, keepdims=True)
        val = jss.logsumexp(val, axis=0, keepdims=True)
        out_ref[i, 0] = val[0].astype(jnp.bfloat16)

    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((4096, 1), jnp.bfloat16),
        grid=(4096,),
        in_specs=(
            pl.BlockSpec((1, 8192), lambda i: (i, 0)),
            pl.no_block_spec,
            pl.BlockSpec((8192,), lambda i: (0,)),
        ),
        out_specs=pl.BlockSpec((1, 1), lambda i: (i, 0)),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel",)),
    )(x, weight, bias)
