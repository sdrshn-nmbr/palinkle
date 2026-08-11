import jax
import jax.numpy as jnp
import pallas as pl
from jax import lax
from jax.xla_client import xla_extension as pltpu

def workload(x, weight, bias):
    """Conv2d + GELU + GlobalAvgPool kernel.
    
    Args:
        x: Input tensor of shape [batch, in_channels, height, width]
        weight: Convolution weights of shape [out_channels, in_channels, kernel_h, kernel_w]
        bias: Bias of shape [out_channels]
    
    Returns:
        Output tensor of shape [batch, out_channels]
    """
    # Transpose x from NHWC to HWCN format
    x_t = jnp.transpose(x, (1, 2, 3, 0))  # [in_channels, height, width, batch]
    
    # Transpose weight from HWIO to IOHW format
    kernel = jnp.transpose(weight, (2, 3, 1, 0))  # [kernel_h, kernel_w, in_channels, out_channels]
    
    # Conv2D with VALID padding
    x_conv = lax.conv_general_dilated(
        x_t, 
        kernel,
        window_strides=(1, 1),
        padding='VALID',
        dimension_numbers=('NHWC', 'HWIO', 'NHWC')
    )
    
    # Add bias (reshaped for broadcasting)
    bias_reshaped = jnp.reshape(bias, (1, 1, 1, -1))
    x_add = x_conv + bias_reshaped
    
    # Apply GELU
    x_gelu = jnp.nn.gelu(x_add)
    
    # Global average pooling over spatial dimensions
    x_out = jnp.mean(x_gelu, axis=(1, 2))
    
    return x_out
