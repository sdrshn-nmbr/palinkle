import jax
import jax.numpy as jnp
import jax.experimental.pallas as pl
import jax.experimental.pallas.tpu as pltpu

def workload(x, weight, linear_bias, gn_weight, gn_bias, bias):
    # Stage 1: Gemm + linear_bias
    def gemm_kernel(x_ref, w_ref, lb_ref, out_ref):
        x = x_ref[...]
        w = w_ref[...]
        lb = lb_ref[...]
        out_ref[...] = jnp.dot(x, w.T) + lb
    
    y = pl.pallas_call(
        gemm_kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        grid=(),
        in_specs=(pl.no_block_spec, pl.no_block_spec, pl.no_block_spec),
        out_specs=pl.no_block_spec,
    )(x, weight, linear_bias)
    
    # Stage 2: GroupNorm
    def gn_kernel(y_ref, gn_w_ref, gn_b_ref, out_ref):
        y = y_ref[...]
        N, C = y.shape
        G = 512
        y_reshaped = y.reshape(N, G, C // G)
        mean = jnp.mean(y_reshaped, axis=2, keepdims=True)
        var = jnp.var(y_reshaped, axis=2, keepdims=True)
        y_norm = (y_reshaped - mean) / jnp.sqrt(var + 1e-5)
        y_back = y_norm.reshape(N, C)
        gn_w = gn_w_ref[...]
        gn_b = gn_b_ref[...]
        out_ref[...] = y_back * gn_w + gn_b
    
    y = pl.pallas_call(
        gn_kernel,
        out_shape=jax.ShapeDtypeStruct(y.shape, y.dtype),
        grid=(),
        in_specs=(pl.no_block_spec, pl.no_block_spec, pl.no_block_spec),
        out_specs=pl.no_block_spec,
    )(y, gn_weight, gn_bias)
    
    # Stage 3: Min + reshape + bias add
    def min_bias_kernel(y_ref, bias_ref, out_ref):
        y = y_ref[...]
        y = jnp.min(y, axis=1, keepdims=True)
        N = y.shape[0]
        y = y.reshape(1, 1, N, 1)
        out_ref[...] = y + bias_ref[...]
    
    out = pl.pallas_call(
        min_bias_kernel,
        out_shape=jax.ShapeDtypeStruct(bias.shape, bias.dtype),
        grid=(),
        in_specs=(pl.no_block_spec, pl.no_block_spec),
        out_specs=pl.no_block_spec,
    )(y, bias)
    
    return out
