import jax
import jax.numpy as jnp
import jax.experimental.pallas as pl
import jax.experimental.pallas.tpu as pltpu

def workload(x, gemm_weight, gemm_bias, gn_weight, gn_bias, multiply_weight):
    # Pallas matmul kernel: compute x @ gemm_weight.T + gemm_bias
    def matmul_kernel(x_ref, w_ref, b_ref, out_ref):
        # Load full tiles; x_ref: (128, 8192), w_ref: (8192, 128), b_ref: (128,)
        x_tile = x_ref[...].astype(jnp.float32)
        w_tile = w_ref[...].astype(jnp.float32)
        b_tile = b_ref[...].astype(jnp.float32)
        out_ref[...] = (jnp.dot(x_tile, w_tile) + b_tile).astype(jnp.bfloat16)

    batch_size = x.shape[0]
    out_features = gemm_weight.shape[1]
    
    # Block specs for tiled matmul
    block_b = 128
    block_o = 128
    
    out_shape = jax.ShapeDtypeStruct(x.shape, x.dtype)
    
    # We need to handle the case where dimensions aren't perfect multiples
    # For this workload they are: 4096/128=32, 8192/128=64
    
    result = pl.pallas_call(
        matmul_kernel,
        out_shape=out_shape,
        grid=(batch_size // block_b, out_features // block_o),
        in_specs=(
            pl.BlockSpec((block_b, gemm_weight.shape[0]), lambda i, j: (i * block_b, 0)),
            pl.BlockSpec((gemm_weight.shape[0], block_o), lambda i, j: (0, j * block_o)),
            pl.BlockSpec((block_o,), lambda i, j: (j * block_o,)),
        ),
        out_specs=pl.BlockSpec((block_b, block_o), lambda i, j: (i * block_b, j * block_o)),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel", "parallel")),
    )(x, gemm_weight.T, gemm_bias)
    
    # GroupNorm
    num_groups = 256
    group_size = out_features // num_groups  # 32
    
    x_grouped = result.reshape(batch_size, num_groups, group_size)
    mean = jnp.mean(x_grouped, axis=-1, keepdims=True)
    var = jnp.var(x_grouped, axis=-1, keepdims=True)
    x_normalized = (x_grouped - mean) / jnp.sqrt(var + 1e-5)
    x = x_normalized.reshape(batch_size, out_features)
    
    # Apply gn_weight and gn_bias
    x = x * gn_weight + gn_bias
    
    # Swish: x * sigmoid(x)
    x = x * jax.nn.sigmoid(x)
    
    # Multiply by multiply_weight
    x = x * multiply_weight
    
    # Swish again
    x = x * jax.nn.sigmoid(x)
    
    return x
