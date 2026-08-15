import jax
import jax.numpy as jnp
import jax.pallas as pl
import jax.pallas as plu
import jax.pallas.tpu as pltpu

def workload(x, weight, conv_bias, gamma, beta):
    """Conv3d + GroupNorm + Mean workload."""
    num_groups = 8
    
    # Transpose x from NCDHW to NDHWC
    x = jnp.transpose(x, (0, 2, 3, 4, 1))
    
    # Transpose weight from OIDHW to DHWIO
    kernel = jnp.transpose(weight, (2, 3, 4, 1, 0))
    
    # Conv3D
    x = jax.lax.conv_general_dilated(
        x, kernel,
        window_strides=(1, 1, 1),
        padding="VALID",
        dimension_numbers=("NDHWC", "DHWIO", "NDHWC")
    )
    
    # Add bias
    x = x + jnp.reshape(conv_bias, (1, 1, 1, 1, -1))
    
    # Transpose back to NCDHW
    x = jnp.transpose(x, (0, 4, 1, 2, 3))
    
    # Get shape
    N, C, D, H, W = x.shape
    G = num_groups
    
    # Reshape for GroupNorm
    x = jnp.reshape(x, (N, G, C // G, D, H, W))
    
    # Compute mean and variance over spatial dimensions
    mean = jnp.mean(x, axis=(2, 3, 4, 5), keepdims=True)
    var = jnp.var(x, axis=(2, 3, 4, 5), keepdims=True)
    
    # Normalize
    x = (x - mean) / jnp.sqrt(var + 1e-5)
    
    # Reshape back
    x = jnp.reshape(x, (N, C, D, H, W))
    
    # Apply gamma and beta
    x = x * jnp.reshape(gamma, (1, -1, 1, 1, 1)) + jnp.reshape(beta, (1, -1, 1, 1, 1))
    
    # Compute mean over channel and spatial dimensions
    x = jnp.mean(x, axis=(1, 2, 3, 4))
    
    return x
