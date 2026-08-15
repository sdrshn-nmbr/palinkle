import jax
import jax.numpy as jnp
import jax.lax as lax
import jax.nn as nn
import pallas as pl
import pallas.triton as pltpu


def workload(x, weight, bias):
    """Conv2d + GELU + GlobalAvgPool kernel."""
    # Transpose x from NCHW to NHWC
    x_t = jnp.transpose(x, (0, 2, 3, 1))  # [128, 256, 256, 8]
    
    # Transpose weight from OIHW to HWIO
    kernel = jnp.transpose(weight, (2, 3, 1, 0))  # [3, 3, 8, 64]
    
    # Conv2d with VALID padding
    conv_out = lax.conv_general_dilated(
        x_t,
        kernel,
        window_strides=(1, 1),
        padding="VALID",
        dimension_numbers=("NHWC", "HWIO", "NHWC")
    )  # [128, 254, 254, 64]
    
    # Add bias (broadcasted)
    bias_reshaped = jnp.reshape(bias, (1, 1, 1, 64))
    out = conv_out + bias_reshaped
    
    # Apply GELU
    out = nn.gelu(out)
    
    # Global average pooling over spatial dimensions (H, W)
    out = jnp.mean(out, axis=(1, 2))  # [128, 64]
    
    return out
