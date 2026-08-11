import jax
import jax.numpy as jnp
import pallas as pl
import pl as pallas_lib
from jax.pallas import lib as pl_lib
from jax.pallas import triton as pltpu

def workload(x, conv_weight, conv_bias, gn_weight, gn_bias):
    """
    Conv2d + GroupNorm + Tanh + HardSwish + ResidualAdd + LogSumExp
    """
    groups = 16
    eps = 1e-5
    
    # Transpose x from NCHW to NHWC
    x_nhwc = jnp.transpose(x, (0, 2, 3, 1))
    
    # Transpose kernel from OIHW to HWIO
    kernel = jnp.transpose(conv_weight, (2, 3, 1, 0))
    
    # Convolution: NHWC * HWIO -> NHWC
    x_conv = jnp.dot(x_nhwc, kernel, dimension_numbers=('NHWC', 'HWIO', 'NHWC'))
    
    # Add bias with reshaping
    x_conv = x_conv + conv_bias.reshape(1, 1, 1, -1)
    
    # Transpose back to NCHW
    x_conv = jnp.transpose(x_conv, (0, 3, 1, 2))
    
    # Get shape and reshape for group norm
    N, C, H, W = x_conv.shape
    x = jnp.reshape(x_conv, (N, groups, C // groups, H, W))
    
    # Group normalization
    mean = jnp.mean(x, axis=(2, 3, 4), keepdims=True)
    var = jnp.var(x, axis=(2, 3, 4), keepdims=True)
    x_norm = (x - mean) / jnp.sqrt(var + eps)
    
    # Scale and shift
    x_norm = x_norm * gn_weight.reshape(1, 1, -1, 1, 1) + gn_bias.reshape(1, 1, -1, 1, 1)
    
    # Tanh
    x_tanh = jnp.tanh(x_norm)
    
    # HardSwish
    x_hard_swish = jnp.minimum(jnp.maximum(x_tanh, 0), 6) * x_tanh / 6
    
    # Residual add (need to reshape x_conv back to match)
    x_conv_reshaped = jnp.reshape(x_conv, (N, groups, C // groups, H, W))
    x_res = x_conv_reshaped + x_hard_swish
    
    # LogSumExp along group axis
    x_logsumexp = jnp.log(jnp.sum(jnp.exp(x_res), axis=1, keepdims=True))
    
    return x_logsumexp
