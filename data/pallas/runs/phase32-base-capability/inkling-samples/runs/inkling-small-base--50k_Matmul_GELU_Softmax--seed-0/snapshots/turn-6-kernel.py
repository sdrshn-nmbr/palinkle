import jax
import jax.numpy as jnp
import jax.experimental.pallas as pl
import jax.experimental.pallas.tpu as pltpu

def workload(x, weight, bias):
    batch_size, in_features = x.shape
    out_features = weight.shape[1]
    
    def kernel(x_ref, w_ref, b_ref, out_ref, acc_ref):
        # Initialize accumulator
        acc_ref[...] = jnp.zeros_like(acc_ref, dtype=jnp.float32)
        block_k = 128
        # Loop over in_features
        for k in range(0, in_features, block_k):
            x_tile = x_ref[:, k:k+block_k].astype(jnp.float32)
            w_tile = w_ref[k:k+block_k, :].astype(jnp.float32)
            acc_ref[...] = acc_ref[...] + jnp.dot(x_tile, w_tile)
        # Add bias
        acc_ref[...] = acc_ref[...] + b_ref[...].astype(jnp.float32)
        # GELU exact
        gelu_val = 0.5 * acc_ref[...] * (1.0 + jnp.erf(acc_ref[...] / jnp.sqrt(2.0)))
        # Softmax over axis=1
        max_val = jnp.max(gelu_val, axis=1, keepdims=True)
        exp_shifted = jnp.exp(gelu_val - max_val)
        sum_exp = jnp.sum(exp_shifted, axis=1, keepdims=True)
        softmax_val = exp_shifted / sum_exp
        out_ref[...] = softmax_val.astype(jnp.bfloat16)
    
    batch_tile = 128
    grid = (batch_size // batch_tile,)
    
    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        grid=grid,
        in_specs=(
            pl.BlockSpec((batch_tile, in_features), lambda i: (i * batch_tile, 0)),
            pl.no_block_spec,
            pl.no_block_spec,
        ),
        out_specs=pl.BlockSpec((batch_tile, out_features), lambda i: (i * batch_tile, 0)),
        scratch_shapes=[pltpu.VMEM((batch_tile, out_features), jnp.float32)],
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel",)),
    )(x, weight, bias)
