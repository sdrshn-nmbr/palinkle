import jax
import jax.numpy as jnp
import jax.pallas as pl
import jax.pallas as pltpu

def workload(x, weight, conv_bias, bias):
    """Conv2D + ReLU + BiasAdd kernel."""
    
    # Transpose x from NCHW to NHWC: [128, 64, 128, 128] -> [128, 128, 128, 64]
    x_nhwc = jnp.transpose(x, (0, 2, 3, 1))
    
    # Transpose weight from HWIO to HWIO format: [128, 64, 3, 3] -> [3, 3, 64, 128]
    kernel = jnp.transpose(weight, (2, 3, 1, 0))
    
    # Conv2D: NHWC input, HWIO kernel, NHWC output
    # Output shape: [128, 126, 126, 128]
    conv_out = jax.lax.conv_general_dilated(
        x_nhwc,
        kernel,
        window_strides=(1, 1),
        padding="VALID",
        dimension_numbers=("NHWC", "HWIO", "NHWC")
    )
    
    # Add conv_bias: reshape to [1, 1, 1, 128] and broadcast
    conv_out = conv_out + jnp.reshape(conv_bias, (1, 1, 1, -1))
    
    # ReLU activation
    relu_out = jax.nn.relu(conv_out)
    
    # Transpose from NHWC to NCHW: [128, 126, 126, 128] -> [128, 128, 126, 126]
    output = jnp.transpose(relu_out, (0, 3, 1, 2))
    
    # Add bias: [128, 128, 126, 126] + [128, 1, 1] -> broadcast
    output = output + bias
    
    return output
