import jax
import jax.numpy as jnp
import jax.random as jrandom
import pallas as pl
import pallas.core as plc
import jax.pallas as pl

def workload(x, conv_weight, conv_bias, gn_weight, gn_bias):
    """Conv2d + GroupNorm + Tanh + HardSwish + ResidualAdd + LogSumExp"""
    
    # Constants
    groups = 16
    eps = 1e-5
    
    # Transpose x from NCHW to NHWC
    # x: [batch, channels, height, width] -> [batch, height, width, channels]
    x_nhwc = jnp.transpose(x, (0, 2, 3, 1))
    
    # Transpose conv_weight from OIHW to IOHW for convolution
    # conv_weight: [out_channels, in_channels, kernel_h, kernel_w] -> [kernel_h, kernel_w, in_channels, out_channels]
    kernel = jnp.transpose(conv_weight, (2, 3, 1, 0))
    
    # Conv2D in NHWC format
    # x_nhwc: [batch, height, width, channels]
    # kernel: [kernel_h, kernel_w, in_channels, out_channels]
    x_conv = jnp.dot(
        x_nhwc,  # Will be handled by conv_general_dilated
        kernel,
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
        padding='VALID',
        window_strides=(1, 1)
    )
    
    # Actually use conv_general_dilated directly
    x_conv = jax.lax.conv_general_dilated(
        x_nhwc,
        conv_weight,
        window_strides=(1, 1),
        padding='VALID',
        dimension_numbers=('NHWC', 'IOHW', 'NHWC')
    )
    
    # Add conv_bias (reshaped for broadcasting)
    # conv_bias: [out_channels] -> [1, 1, 1, out_channels]
    conv_bias_reshaped = jnp.reshape(conv_bias, (1, 1, 1, -1))
    x_conv = x_conv + conv_bias_reshaped
    
    # Transpose from NHWC to NCHW
    # x_conv: [batch, height, width, channels] -> [batch, channels, height, width]
    x_conv = jnp.transpose(x_conv, (0, 3, 1, 2))
    
    # Get shapes
    N, C, H, W = x_conv.shape
    
    # Reshape for group norm: [batch, groups, channels/groups, height, width]
    # C = 64, groups = 16, so channels/groups = 4
    x = jnp.reshape(x_conv, (N, groups, C // groups, H, W))
    
    # GroupNorm: compute mean and variance over axes [2, 3, 4]
    mean = jnp.mean(x, axis=(2, 3, 4), keepdims=True)
    var = jnp.var(x, axis=(2, 3, 4), keepdims=True)
    
    # Normalize
    x_norm = (x - mean) / jnp.sqrt(var + eps)
    
    # Scale and shift
    # gn_weight: [channels] -> [1, 1, channels/groups, 1, 1]
    # gn_bias: [channels] -> [1, 1, channels/groups, 1, 1]
    gn_weight_reshaped = jnp.reshape(gn_weight, (1, 1, C // groups, 1, 1))
    gn_bias_reshaped = jnp.reshape(gn_bias, (1, 1, C // groups, 1, 1))
    x_norm = x_norm * gn_weight_reshaped + gn_bias_reshaped
    
    # Reshape back to [batch, channels, height, width]
    x_norm = jnp.reshape(x_norm, (N, C, H, W))
    
    # Apply tanh
    x_tanh = jnp.tanh(x_norm)
    
    # HardSwish: x * hard_sigmoid(x + 3) / 6
    # hard_sigmoid(x) = max(0, min(x + 3, 6)) / 6
    x_hard_swish = x_tanh * jnp.clip(x_tanh + 3, 0, 6) / 6
    
    # Residual add
    x_res = x_conv + x_hard_swish
    
    # LogSumExp over axis 1 (groups dimension was removed, now over channels)
    # Actually need to reshape back first
    x_res = jnp.reshape(x_res, (N, groups, C // groups, H, W))
    x_logsumexp = jax.scipy.special.logsumexp(x_res, axis=1, keepdims=True)
    
    # Final output shape: [batch, 1, height, width]
    return jnp.reshape(x_logsumexp, (N, 1, H, W))
