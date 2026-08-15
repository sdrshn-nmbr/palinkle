import jax
import jax.numpy as jnp
import jax.pallas as pl
import jax.pallas as plp
import jax.interpreters.pallas as pallas
from jax.interpreters import pallas as pl
from jax.pallas import lib as pllib
import jax.numpy as jnp

def workload(x, weight, bias):
    """
    Matmul + AvgPool1d + GELU + Scale + Max
    
    x: [4096, 8192] bfloat16
    weight: [8192, 8192] bfloat16
    bias: [8192] bfloat16
    
    Output: [4096] bfloat16
    """
    pool_kernel_size = 16
    scale_factor = 2.0
    
    # Matmul: x @ weight + bias -> [4096, 8192]
    matmul_out = jnp.dot(x, weight) + bias
    
    # Expand dims on axis 1 -> [4096, 1, 8192]
    expanded = jnp.expand_dims(matmul_out, axis=1)
    
    # Avg pool with kernel size 16 -> [4096, 1, 512]
    # reduce_window with VALID padding, kernel [1, 1, 16], stride [1, 1, 16]
    pooled = jax.lax.reduce_window(
        expanded,
        init_value=0.0,
        computation=jax.lax.add,
        window_dimensions=(1, 1, pool_kernel_size),
        window_strides=(1, 1, pool_kernel_size),
        padding="VALID"
    ) / pool_kernel_size
    
    # Squeeze on axis 1 -> [4096, 512]
    squeezed = jnp.squeeze(pooled, axis=1)
    
    # GELU activation -> [4096, 512]
    gelu_out = jax.nn.gelu(squeezed)
    
    # Scale by 2.0 -> [4096, 512]
    scaled = gelu_out * scale_factor
    
    # Max along axis 1 -> [4096]
    result = jnp.max(scaled, axis=1)
    
    return result
