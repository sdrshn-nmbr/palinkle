import jax
import jax.numpy as jnp
import jax.lax as lax
import jax.nn as nn
import pallas as pl
import pytpu

def workload(x, conv_weight, conv_bias, bn_weight, bn_bias):
    """Conv2d + Mish activation + BatchNorm kernel."""
    
    # Constants
    eps = 1e-05
    
    # Transpose x from NHWC to NCHW: [64, 64, 128, 128] -> [64, 128, 128, 128]
    x_t = jnp.transpose(x, (0, 3, 1, 2))
    
    # Transpose weight from HWIO to OIHW: [128, 64, 3, 3] -> [3, 3, 64, 128]
    weight_t = jnp.transpose(conv_weight, (2, 3, 1, 0))
    
    # Conv2d with VALID padding, stride 1
    conv_out = lax.conv_general_dilated(
        x_t,
        weight_t,
        window_strides=(1, 1),
        padding="VALID",
        dimension_numbers=("NCHW", "OIHW", "NCHW")
    )
    
    # Add bias: reshape to [1, 128, 1, 1]
    bias_reshaped = jnp.reshape(conv_bias, (1, 128, 1, 1))
    x = conv_out + bias_reshaped
    
    # Transpose back to NHWC: [64, 128, 126, 126] -> [64, 126, 126, 128]
    x = jnp.transpose(x, (0, 2, 3, 1))
    
    # Mish activation: x * tanh(softplus(x))
    softplus_x = nn.softplus(x)
    x = jnp.multiply(jnp.tanh(softplus_x), x)
    
    # BatchNorm: compute mean and variance over axes [0, 2, 3]
    mean = jnp.mean(x, axis=(0, 2, 3), keepdims=True)
    var = jnp.mean(jnp.power(jnp.subtract(x, mean), 2), axis=(0, 2, 3), keepdims=True)
    
    # Reshape bn_weight and bn_bias to [1, 1, 1, 128]
    w = jnp.reshape(bn_weight, (1, 1, 1, 128))
    b = jnp.reshape(bn_bias, (1, 1, 1, 128))
    
    # BatchNorm: (x - mean) / sqrt(var + eps) * w + b
    x = jnp.add(
        jnp.multiply(
            jnp.divide(
                jnp.subtract(x, mean),
                jnp.sqrt(jnp.add(var, eps))
            ),
            w
        ),
        b
    )
    
    return x
