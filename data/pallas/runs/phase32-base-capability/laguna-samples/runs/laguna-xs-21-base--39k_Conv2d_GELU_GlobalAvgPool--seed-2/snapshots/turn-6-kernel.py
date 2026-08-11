import jax
import jax.numpy as jnp
import pallas as pl
import pallas.core as plc
import jax.pallas as plu
import jax.interpreters.pallas as pallas
from jax.interpreters import pallas as pl
import jax.numpy as jnp

def workload(x, weight, bias):
    """Conv2d + GELU + GlobalAvgPool kernel."""
    # Transpose x from NCHW to NHWC
    x_t = x.transpose(0, 2, 3, 1)  # (128, 256, 256, 8)
    
    # Transpose weight from OIHW to HWIO
    kernel = weight.transpose(2, 3, 1, 0)  # (3, 3, 8, 64)
    
    # Conv2D: NHWC * HWIO -> NHWC
    conv_out = jax.lax.conv_general_dilated(
        x_t, kernel,
        window_strides=(1, 1),
        padding='VALID',
        dimension_numbers=('NHWC', 'HWIO', 'NHWC')
    )  # (128, 254, 254, 64)
    
    # Add bias with reshaping for broadcasting
    bias_reshaped = bias.reshape(1, 1, 1, 64)  # (1, 1, 1, 64)
    x_add = conv_out + bias_reshaped
    
    # Apply GELU activation
    x_gelu = jax.nn.gelu(x_add)
    
    # Global Average Pooling over H and W axes
    result = jnp.mean(x_gelu, axis=(1, 2))  # (128, 64)
    
    return result
