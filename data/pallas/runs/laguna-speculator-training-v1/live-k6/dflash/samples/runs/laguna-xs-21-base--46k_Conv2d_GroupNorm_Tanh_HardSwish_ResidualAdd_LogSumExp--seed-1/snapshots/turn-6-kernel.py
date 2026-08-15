import jax
import jax.numpy as jnp
import jax.pallas as pl
import jax.pallas as plu
import jax.interpre.pallas as pj
import jax.lax as lax

def workload(x, conv_weight, conv_bias, gn_weight, gn_bias):
    """
    Conv2d + GroupNorm + Tanh + HardSwish + ResidualAdd + LogSumExp
    
    Input shapes:
    - x: [128, 8, 128, 128] (NCHW)
    - conv_weight: [64, 8, 3, 3]
    - conv_bias: [64]
    - gn_weight: [64]
    - gn_bias: [64]
    
    Output shape: [128, 1, 126, 126]
    """
    groups = 16
    eps = 1e-5
    
    # Step 1: Transpose x from NCHW to NHWC
    x_nhwc = jnp.transpose(x, (0, 2, 3, 1))
    
    # Step 2: Transpose conv_weight from [64, 8, 3, 3] to [3, 3, 8, 64] (HWIO)
    kernel = jnp.transpose(conv_weight, (2, 3, 1, 0))
    
    # Step 3: Conv2d with valid padding
    x_conv = lax.conv_general_dilated(
        x_nhwc,
        kernel,
        window_strides=(1, 1),
        padding="VALID",
        dimension_numbers=("NHWC", "HWIO", "NHWC")
    )
    
    # Step 4: Add bias (broadcasted)
    x_conv = x_conv + jnp.reshape(conv_bias, (1, 1, 1, 64))
    
    # Step 5: Transpose from NHWC to NCHW
    x_conv = jnp.transpose(x_conv, (0, 3, 1, 2))
    
    # Step 6: Reshape for GroupNorm: [N, groups, C//groups, H, W]
    N, C, H, W = x_conv.shape
    x = jnp.reshape(x_conv, (N, groups, C // groups, H, W))
    
    # Step 7: GroupNorm - compute mean and variance over spatial dimensions
    mean = jnp.mean(x, axis=(2, 3, 4), keepdims=True)
    var = jnp.var(x, axis=(2, 3, 4), keepdims=True)
    
    # Step 8: Normalize
    x = (x - mean) / jnp.sqrt(var + eps)
    
    # Step 9: Reshape back to NCHW
    x = jnp.reshape(x, (N, C, H, W))
    
    # Step 10: Scale and shift (gamma and beta)
    x_norm = x * jnp.reshape(gn_weight, (1, C, 1, 1)) + jnp.reshape(gn_bias, (1, C, 1, 1))
    
    # Step 11: Tanh
    x_tanh = jnp.tanh(x_norm)
    
    # Step 12: HardSwish
    x_hard_swish = (x_tanh * jnp.minimum(jnp.maximum(x_tanh + 3, 0), 6)) / 6
    
    # Step 13: Residual add with conv output
    x_res = x_conv + x_hard_swish
    
    # Step 14: LogSumExp along axis 1
    x_logsumexp = jax.scipy.special.logsumexp(x_res, axis=1, keepdims=True)
    
    return x_logsumexp
