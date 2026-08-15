import jax
import jax.numpy as jnp
import jax.lax as lax
import pallas as pl
import jax.pallas.tpu as pltpu

def workload(x, weight, bias):
    """Conv3d + Mish + Tanh kernel implemented as a Pallas kernel."""
    
    # Transpose x from [16, 32, 32, 64, 64] to [16, 64, 32, 32, 32]
    x = jnp.transpose(x, (0, 4, 1, 2, 3))
    
    # Transpose weight from [64, 32, 3, 3, 3] to [3, 3, 3, 32, 64]
    kernel = jnp.transpose(weight, (2, 3, 4, 1, 0))
    
    # Conv3D with NDHWC/DHWIO/NDHWC format
    x = lax.conv_general_dilated(
        x,
        kernel,
        window_strides=(1, 1, 1),
        padding=((0, 0), (0, 0), (0, 0)),
        dimension_numbers=("NDHWC", "DHWIO", "NDHWC")
    )
    
    # Add bias
    x = x + jnp.reshape(bias, (1, 64, 1, 1, 1))
    
    # Mish: x * tanh(log(1 + exp(x)))
    x = x * jnp.tanh(jnp.log(1 + jnp.exp(x)))
    
    # Tanh
    x = jnp.tanh(x)
    
    # Final transpose from [16, 62, 30, 30, 64] to [16, 64, 30, 62, 62]
    x = jnp.transpose(x, (0, 4, 1, 2, 3))
    
    return x
