import jax
import jax.numpy as jnp
import jax.pallas as pl
import jax.pallas as plp
import jax.interpreters.tpu as pltpu


def workload(x, weight, bias):
    """Gemm + Scaling + Hardtanh + GELU kernel."""
    batch_size, in_features = x.shape
    _, out_features = weight.shape
    
    block_size = 128  # Block size for tiling
    
    def kernel(x_ref, weight_ref, bias_ref, out_ref):
        # Get program IDs for grid layout
        m = pl.program_id(0)  # batch dimension
        n = pl.program_id(1)  # output feature dimension
        
        # Accumulator in float32 for better precision
        acc = jnp.zeros(out_features, dtype=jnp.float32)
        
        # Matmul: accumulate over input features
        for k in range(in_features // block_size):
            x_block = x_ref[m, k * block_size:(k + 1) * block_size]
            weight_block = weight_ref[k * block_size:(k + 1) * block_size, n * block_size:(n + 1) * block_size]
            acc = acc + jnp.dot(x_block.astype(jnp.float32), weight_block.astype(jnp.float32))
        
        # Add bias
        acc = acc + bias_ref[n * block_size:(n + 1) * block_size].astype(jnp.float32)
        
        # Scale by 0.5
        acc = acc * 0.5
        
        # Hardtanh with min=-2, max=2
        acc = jnp.clip(acc, -2.0, 2.0)
        
        # GELU: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        gelu_factor = 0.5 * jnp.sqrt(2.0 / jnp.pi)
        gelu_inner = gelu_factor * (acc + 0.044715 * jnp.power(acc, 3))
        acc = 0.5 * acc * (1.0 + jnp.tanh(gelu_inner))
        
        # Write output
        out_ref[m, n * block_size:(n + 1) * block_size] = acc.astype(x.dtype)
    
    # Grid dimensions
    grid = (batch_size, out_features // block_size)
    
    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((batch_size, out_features), x.dtype),
        grid=grid,
        in_specs=(
            pl.BlockSpec((batch_size, in_features), lambda m, n: (m, 0)),
            pl.BlockSpec((in_features, out_features), lambda m, n: (0, n * block_size)),
            pl.BlockSpec((out_features,), lambda m, n: (n * block_size,)),
        ),
        out_specs=pl.BlockSpec((batch_size, out_features), lambda m, n: (m, 0)),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel")
        ),
    )(x, weight, bias)
