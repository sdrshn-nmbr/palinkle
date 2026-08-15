import jax
import jax.numpy as jnp
import jax.pallas as pl
import jax.pallas as plp
import jax.random as random

def workload(x, weight, conv_bias, gamma, beta):
    """Conv3d + GroupNorm + Mean workload."""
    
    # Constants
    num_groups = 8
    
    # Step 1: Transpose x from NCDHW to NDHWC
    x = jnp.transpose(x, (0, 2, 3, 4, 1))
    
    # Step 2: Transpose weight from OCDHW to DHWIO
    kernel = jnp.transpose(weight, (2, 3, 4, 1, 0))
    
    # Step 3: Conv3D with VALID padding
    x = jax.lax.conv_general_dilated(
        x, kernel,
        window_strides=(1, 1, 1),
        padding='VALID',
        dimension_numbers=('NDHWC', 'DHWIO', 'NDHWC')
    )
    
    # Step 4: Add bias
    x = x + conv_bias.reshape(1, 1, 1, 1, 24)
    
    # Step 5: Transpose back to NCDHW
    x = jnp.transpose(x, (0, 4, 1, 2, 3))
    
    # Get shape
    N, C, D, H, W = x.shape
    G = num_groups
    
    # Step 6: Reshape for GroupNorm
    x = x.reshape(N, G, C // G, D, H, W)
    
    # Step 7: Compute mean and variance over spatial dimensions
    mean = jnp.mean(x, axis=(2, 3, 4, 5), keepdims=True)
    var = jnp.var(x, axis=(2, 3, 4, 5), keepdims=True)
    
    # Step 8: Normalize
    x = (x - mean) / jnp.sqrt(var + 1e-5)
    
    # Step 9: Reshape back to NCDHW
    x = x.reshape(N, C, D, H, W)
    
    # Step 10: Scale and shift
    x = x * gamma.reshape(1, C, 1, 1, 1) + beta.reshape(1, C, 1, 1, 1)
    
    # Step 11: Final mean reduction over spatial dimensions
    x = jnp.mean(x, axis=(1, 2, 3, 4))
    
    return x
