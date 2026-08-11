import jax
import jax.numpy as jnp
import jax.experimental.pallas as pl
import jax.experimental.pallas.tpu as pltpu

def workload(x, gemm_weight, gemm_bias, bn_weight, bn_bias):
    block_f = 128
    out_f = x.shape[1]  # 8192
    grid_f = out_f // block_f  # 64
    
    def kernel(x_ref, gemm_weight_ref, gemm_bias_ref, bn_weight_ref, bn_bias_ref, out_ref):
        x_val = x_ref[...]
        w_tile = gemm_weight_ref[...]
        b_tile = gemm_bias_ref[...]
        bn_w = bn_weight_ref[...]
        bn_b = bn_bias_ref[...]
        
        # Gemm for this feature tile
        gemm = jnp.dot(x_val, w_tile.T) + b_tile
        
        eps = 1e-05
        mean = jnp.mean(gemm, axis=0, keepdims=True)
        var = jnp.mean((gemm - mean) ** 2, axis=0, keepdims=True)
        bn = (gemm - mean) / jnp.sqrt(var + eps) * bn_w + bn_b
        
        gelu = jax.nn.gelu(bn)
        relu = jax.nn.relu(gelu)
        
        out_ref[...] = relu
    
    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        grid=(grid_f,),
        in_specs=(
            pl.no_block_spec,
            pl.BlockSpec((block_f, x.shape[1]), lambda i: (i, 0)),
            pl.BlockSpec((block_f,), lambda i: (i,)),
            pl.BlockSpec((block_f,), lambda i: (i,)),
            pl.BlockSpec((block_f,), lambda i: (i,)),
        ),
        out_specs=pl.BlockSpec((x.shape[0], block_f), lambda i: (0, i)),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel",)),
    )(x, gemm_weight, gemm_bias, bn_weight, bn_bias)
