import jax
import jax.numpy as jnp
import jax.pallas as pl
import jax.pallas as plp
import jax.interpreters.tpu as pltpu


def matmul_subtract_multiply_relu_kernel(x_ref, weight_ref, bias_ref, out_ref):
    """Pallas kernel for matmul + subtract + multiply + relu."""
    # Get grid dimensions
    m = pl.program_id(0)
    n = pl.program_id(1)
    
    # Block size for tiling
    block_k = 128
    
    # Initialize accumulator in float32
    acc = jnp.zeros(x_ref.shape[1], dtype=jnp.float32)
    
    # Matmul: x[m, :] @ weight[:, n] + bias[n]
    # x_ref has shape [block_m, in_features]
    # weight_ref has shape [in_features, block_n]
    # bias_ref has shape [block_n]
    
    # Perform matmul with accumulation in float32
    for k in range(0, x_ref.shape[1], block_k):
        x_block = x_ref[:, k:k+block_k].astype(jnp.float32)
        weight_block = weight_ref[k:k+block_k, :].astype(jnp.float32)
        acc = acc + jnp.dot(x_block, weight_block)
    
    # Add bias
    acc = acc + bias_ref.astype(jnp.float32)
    
    # Subtract 2.0
    acc = acc - 2.0
    
    # Multiply by 1.5
    acc = acc * 1.5
    
    # ReLU
    acc = jnp.maximum(acc, 0.0)
    
    # Store result as bfloat16
    out_ref[...] = acc.astype(jnp.bfloat16)


def workload(x, weight, bias):
    """Workload: matmul + subtract + multiply + relu."""
    batch_size = x.shape[0]
    in_features = x.shape[1]
    out_features = weight.shape[1]
    
    # Block sizes for TPU
    block_m = 128
    block_n = 128
    block_k = 128
    
    # Grid dimensions
    grid_m = (batch_size + block_m - 1) // block_m
    grid_n = (out_features + block_n - 1) // block_n
    
    return pl.pallas_call(
        matmul_subtract_multiply_relu_kernel,
        out_shape=jax.ShapeDtypeStruct((batch_size, out_features), jnp.bfloat16),
        grid=(grid_m, grid_n),
        in_specs=(
            pl.BlockSpec((block_m, in_features), lambda m, n: (m * block_m, 0)),
            pl.BlockSpec((in_features, block_n), lambda m, n: (0, n * block_n)),
            pl.BlockSpec((out_features,), lambda m, n: (n * block_n,)),
        ),
        out_specs=pl.BlockSpec((block_m, block_n), lambda m, n: (m * block_m, n * block_n)),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel")
        ),
    )(x, weight, bias)
