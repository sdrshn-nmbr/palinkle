import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
import jax.experimental.pallas.tpu as pltpu

BATCH_BLOCK = 128
FEATURE_BLOCK = 128

def gemm_kernel(x_ref, w_ref, b_ref, out_ref):
    x_local = x_ref[...].astype(jnp.float32)
    w_local = w_ref[...].astype(jnp.float32)
    b_local = b_ref[...].astype(jnp.float32)
    # x_local: (B, K), w_local: (N, K) -> dot(x, w.T) = (B, N)
    acc = jnp.dot(x_local, w_local.T)
    acc += b_local
    out_ref[...] = acc.astype(jnp.bfloat16)

def bn_kernel(x_ref, mean_ref, var_ref, bn_w_ref, bn_b_ref, out_ref):
    x_local = x_ref[...].astype(jnp.float32)
    mean_local = mean_ref[...].astype(jnp.float32)
    var_local = var_ref[...].astype(jnp.float32)
    bn_w_local = bn_w_ref[...].astype(jnp.float32)
    bn_b_local = bn_b_ref[...].astype(jnp.float32)
    eps = 1e-05
    # batchnorm
    x_norm = (x_local - mean_local) / jnp.sqrt(var_local + eps)
    x_norm = x_norm * bn_w_local + bn_b_local
    # gelu exact
    gelu = 0.5 * x_norm * (1.0 + jnp.erf(x_norm / jnp.sqrt(2.0)))
    # relu
    out = jnp.maximum(gelu, 0.0)
    out_ref[...] = out.astype(jnp.bfloat16)

def workload(x, gemm_weight, gemm_bias, bn_weight, bn_bias):
    batch_size, in_features = x.shape
    out_features = gemm_weight.shape[1]
    # Gemm via Pallas
    gemm_out = pl.pallas_call(
        gemm_kernel,
        out_shape=jax.ShapeDtypeStruct((batch_size, out_features), jnp.bfloat16),
        grid=(batch_size // BATCH_BLOCK, out_features // FEATURE_BLOCK),
        in_specs=(
            pl.BlockSpec((BATCH_BLOCK, in_features), lambda i, j: (i * BATCH_BLOCK, 0)),
            pl.BlockSpec((FEATURE_BLOCK, in_features), lambda i, j: (j * FEATURE_BLOCK, 0)),
            pl.BlockSpec((FEATURE_BLOCK,), lambda i, j: (j * FEATURE_BLOCK,)),
        ),
        out_specs=pl.BlockSpec((BATCH_BLOCK, FEATURE_BLOCK), lambda i, j: (i * BATCH_BLOCK, j * FEATURE_BLOCK)),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel", "parallel")),
    )(x, gemm_weight, gemm_bias)
    # BatchNorm stats
    mean = jnp.mean(gemm_out, axis=0, keepdims=True)
    var = jnp.mean((gemm_out - mean) ** 2, axis=0, keepdims=True)
    # BatchNorm + GELU + ReLU via Pallas
    final = pl.pallas_call(
        bn_kernel,
        out_shape=jax.ShapeDtypeStruct((batch_size, out_features), jnp.bfloat16),
        grid=(batch_size // BATCH_BLOCK, out_features // FEATURE_BLOCK),
        in_specs=(
            pl.BlockSpec((BATCH_BLOCK, FEATURE_BLOCK), lambda i, j: (i * BATCH_BLOCK, j * FEATURE_BLOCK)),
            pl.BlockSpec((1, FEATURE_BLOCK), lambda i, j: (0, j * FEATURE_BLOCK)),
            pl.BlockSpec((1, FEATURE_BLOCK), lambda i, j: (0, j * FEATURE_BLOCK)),
            pl.BlockSpec((FEATURE_BLOCK,), lambda i, j: (j * FEATURE_BLOCK,)),
            pl.BlockSpec((FEATURE_BLOCK,), lambda i, j: (j * FEATURE_BLOCK,)),
        ),
        out_specs=pl.BlockSpec((BATCH_BLOCK, FEATURE_BLOCK), lambda i, j: (i * BATCH_BLOCK, j * FEATURE_BLOCK)),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel", "parallel")),
    )(gemm_out, mean, var, bn_weight, bn_bias)
    return final
