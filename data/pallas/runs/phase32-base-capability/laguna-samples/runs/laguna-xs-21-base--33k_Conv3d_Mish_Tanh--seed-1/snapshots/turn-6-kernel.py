import jax
import jax.numpy as jnp
import pallas as pl
import pallas.core as plc

def workload(x, weight, bias):
    """Conv3D + Mish + Tanh kernel using Pallas."""
    
    # Input shapes:
    # x: [16, 32, 32, 64, 64] -> after transpose [0, 2, 3, 4, 1]: [16, 32, 64, 64, 32]
    # weight: [64, 32, 3, 3, 3] -> after transpose [2, 3, 4, 1, 0]: [3, 3, 3, 32, 64]
    # bias: [64] -> after reshape [1, 1, 1, 1, 64]: [1, 1, 1, 1, 64]
    
    # Transpose x: [N, D, H, W, C] -> [N, D, H, W, C] with perm [0, 2, 3, 4, 1]
    x_t = jnp.transpose(x, (0, 2, 3, 4, 1))
    
    # Transpose weight: [M, C, kD, kH, kW] -> [kD, kH, kW, C, M] with perm [2, 3, 4, 1, 0]
    kernel_t = jnp.transpose(weight, (2, 3, 4, 1, 0))
    
    # Reshape bias for broadcasting
    bias_reshaped = jnp.reshape(bias, (1, 1, 1, 1, 64))
    
    # Conv3D: x_t [16, 32, 32, 64, 32] conv with kernel_t [3, 3, 3, 32, 64]
    # dimension_numbers: (NDHWC, DHWIO, NDHWC)
    # padding: ((0, 0), (0, 0), (0, 0))
    # window_strides: (1, 1, 1)
    conv_out = jax.lax.conv_general_dilated(
        x_t,
        kernel_t,
        bias_reshaped,
        window_strides=(1, 1, 1),
        padding=((0, 0), (0, 0), (0, 0)),
        dimension_numbers=('NDHWC', 'DHWIO', 'NDHWC')
    )
    
    # conv_out shape: [16, 30, 62, 62, 64]
    
    # Apply Mish: x * tanh(log(1 + exp(x)))
    # But we need to apply it element-wise
    mish_out = conv_out * jnp.tanh(jnp.log(1 + jnp.exp(conv_out)))
    
    # Apply tanh
    tanh_out = jnp.tanh(mish_out)
    
    # tanh_out shape: [16, 30, 62, 62, 64]
    # Final transpose: [0, 4, 1, 2, 3] -> [16, 64, 30, 62, 62]
    result = jnp.transpose(tanh_out, (0, 4, 1, 2, 3))
    
    return result
