import jax
import jax.numpy as jnp
import jax.pallas as pl
import jax.pallas as plp
import jax.interpreters.pallas as pallas
from jax.interpreters import pallas as pl
from jax.pallas import lib as pllib
import jax.lax as lax

def workload(x, weight, bias):
    """Conv2d + GELU + GlobalAvgPool kernel."""
    
    # Input shapes:
    # x: [128, 8, 256, 256] (NCHW)
    # weight: [64, 8, 3, 3] (OIHW)
    # bias: [64]
    
    # Output shape: [128, 64]
    
    # Transpose to NHWC and HWIO formats
    x_nhwc = jnp.transpose(x, (0, 2, 3, 1))  # [128, 256, 256, 8]
    kernel_hwio = jnp.transpose(weight, (2, 3, 1, 0))  # [3, 3, 8, 64]
    
    # Conv2d
    conv_out = lax.conv_general_dilated(
        x_nhwc,
        kernel_hwio,
        window_strides=(1, 1),
        padding='VALID',
        dimension_numbers=('NHWC', 'HWIO', 'NHWC')
    )  # [128, 256, 256, 64]
    
    # Add bias
    bias_reshaped = jnp.reshape(bias, (1, 1, 1, 64))
    x = conv_out + bias_reshaped
    
    # GELU
    x = jax.nn.gelu(x)
    
    # Global average pool over spatial dimensions (1, 2)
    x = jnp.mean(x, axis=(1, 2))
    
    return x
