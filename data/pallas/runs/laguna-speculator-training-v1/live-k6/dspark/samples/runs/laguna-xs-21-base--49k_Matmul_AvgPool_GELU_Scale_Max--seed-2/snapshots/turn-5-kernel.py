import jax
import jax.numpy as jnp
import jax.pallas as pl
import jax.pallas as plp
import jax.interpreters.mlir as mlir
from jax.interpreters import xla
from jax import xla as jax_xla
import jax.lax as lax

def workload(x, weight, bias):
    """
    Matmul + AvgPool1d + GELU + Scale + Max
    
    x: [4096, 8192] bfloat16
    weight: [8192, 8192] bfloat16
    bias: [8192] bfloat16
    output: [4096] bfloat16
    """
    pool_kernel_size = 16
    scale_factor = 2.0
    
    # Matmul: x @ weight + bias
    # Result shape: [4096, 8192]
    matmul_result = jnp.dot(x, weight) + bias
    
    # Expand dims on axis 1: [4096, 1, 8192]
    expanded = jnp.expand_dims(matmul_result, axis=1)
    
    # AvgPool1d with kernel size 16, stride 16, VALID padding
    # Window dimensions: [1, 1, 16], strides: [1, 1, 16]
    # Result shape: [4096, 1, 512]
    pooled = lax.reduce_window(
        expanded,
        init_value=0.0,
        computation=lax.add,
        window_dimensions=(1, 1, pool_kernel_size),
        window_strides=(1, 1, pool_kernel_size),
        padding="VALID"
    ) / pool_kernel_size
    
    # Squeeze on axis 1: [4096, 512]
    squeezed = jnp.squeeze(pooled, axis=1)
    
    # GELU activation
    gelu_result = jax.nn.gelu(squeezed)
    
    # Scale by 2.0
    scaled = gelu_result * scale_factor
    
    # Max along axis 1: [4096]
    result = jnp.max(scaled, axis=1)
    
    return result
