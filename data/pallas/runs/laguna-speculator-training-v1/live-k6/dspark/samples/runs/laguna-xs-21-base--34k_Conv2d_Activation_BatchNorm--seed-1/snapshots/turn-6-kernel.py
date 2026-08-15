import jax
import jax.numpy as jnp
import jax.lax as lax
import jax.nn as nn
import pallas as pl
import pallas.triton as pltpu


def workload(x, conv_weight, conv_bias, bn_weight, bn_bias):
    """Conv2d + Mish activation + BatchNorm kernel."""
    
    # Transpose x from NCHW to NHWC
    x_nhwc = jnp.transpose(x, (0, 2, 3, 1))
    
    # Transpose conv_weight from HWIO to HWIO (already in correct format for conv_general_dilated)
    # Original: [out_channels, in_channels, kernel_h, kernel_w] = [128, 64, 3, 3]
    # For NHWC input and NHWC output, we need HWIO format: [kernel_h, kernel_w, in_channels, out_channels]
    weight_hwio = jnp.transpose(conv_weight, (2, 3, 1, 0))
    
    # Conv2D: NHWC input, HWIO weight, NHWC output
    conv_out = lax.conv_general_dilated(
        x_nhwc,
        weight_hwio,
        window_strides=(1, 1),
        padding="VALID",
        dimension_numbers=("NHWC", "HWIO", "NHWC")
    )
    
    # Add bias (reshaped to broadcast)
    bias_reshaped = jnp.reshape(conv_bias, (1, 1, 1, -1))
    x = conv_out + bias_reshaped
    
    # Transpose for activation: [64, 126, 126, 128] -> [64, 128, 126, 126]
    x = jnp.transpose(x, (0, 3, 1, 2))
    
    # Mish activation: tanh(softplus(x)) * x
    softplus_x = nn.softplus(x)
    x = jnp.tanh(softplus_x) * x
    
    # BatchNorm: compute mean and variance over axes (0, 2, 3)
    eps = 1e-5
    mean = jnp.mean(x, axis=(0, 2, 3), keepdims=True)
    var = jnp.mean((x - mean) ** 2, axis=(0, 2, 3), keepdims=True)
    
    # Reshape bn_weight and bn_bias for broadcasting
    w = jnp.reshape(bn_weight, (1, -1, 1, 1))
    b = jnp.reshape(bn_bias, (1, -1, 1, 1))
    
    # Apply batch normalization
    x = (x - mean) / jnp.sqrt(var + eps) * w + b
    
    return x


def _conv2d_mish_bn_kernel(
    x_ref,
    weight_ref,
    conv_bias_ref,
    bn_weight_ref,
    bn_bias_ref,
    out_ref,
    eps=1e-5,
):
    """Pallas kernel for Conv2d + Mish + BatchNorm."""
    
    # Get program IDs for parallel execution
    b = pl.program_id(0)  # batch dimension
    c = pl.program_id(1)  # output channel dimension
    h = pl.program_id(2)  # height dimension
    w = pl.program_id(3)  # width dimension
    
    # Block sizes
    block_b = 8
    block_c = 128
    block_h = 126
    block_w = 128
    
    # Read input data
    # x is [64, 64, 128, 128] in NCHW format
    # We need to transpose to NHWC for convolution
    
    # For simplicity, we'll use a more direct approach
    # Read the entire input slice for this block
    
    # Read input x in NCHW format
    x_nchw = x_ref[b * block_b : (b + 1) * block_b, 
                   :, :, :]
    
    # Transpose to NHWC
    x_nhwc = jnp.transpose(x_nchw, (0, 2, 3, 1))
    
    # Read weight in HWIO format
    weight_hwio = weight_ref[:, :, :, :]
    
    # Perform convolution
    conv_out = lax.conv_general_dilated(
        x_nhwc,
        weight_hwio,
        window_strides=(1, 1),
        padding="VALID",
        dimension_numbers=("NHWC", "HWIO", "NHWC")
    )
    
    # Add bias
    bias = conv_bias_ref[:]
    conv_out = conv_out + jnp.reshape(bias, (1, 1, 1, -1))
    
    # Transpose for activation
    x = jnp.transpose(conv_out, (0, 3, 1, 2))
    
    # Mish activation
    softplus_x = nn.softplus(x)
    x = jnp.tanh(softplus_x) * x
    
    # BatchNorm
    mean = jnp.mean(x, axis=(0, 2, 3), keepdims=True)
    var = jnp.mean((x - mean) ** 2, axis=(0, 2, 3), keepdims=True)
    
    w = jnp.reshape(bn_weight_ref[:], (1, -1, 1, 1))
    b = jnp.reshape(bn_bias_ref[:], (1, -1, 1, 1))
    
    x = (x - mean) / jnp.sqrt(var + eps) * w + b
    
    # Write output
    out_ref[b, c, h, w] = x[b, c, h, w]


def workload_pallas(x, conv_weight, conv_bias, bn_weight, bn_bias):
    """Conv2d + Mish activation + BatchNorm using Pallas kernel."""
    
    # Output shape: [64, 128, 126, 126]
    out_shape = jax.ShapeDtypeStruct((64, 128, 126, 126), x.dtype)
    
    # Grid: 64 batches, 128 channels, 126 height, 126 width
    # Use block sizes that are multiples of 8 for bf16
    grid = (8, 128, 126, 128)  # Simplified grid
    
    return pl.pallas_call(
        _conv2d_mish_bn_kernel,
        out_shape=out_shape,
        grid=grid,
        in_specs=(
            pl.BlockSpec((64, 64, 128, 128), lambda b, c, h, w: (b, 0, 0, 0)),  # x
            pl.BlockSpec((128, 64, 3, 3), lambda b, c, h, w: (0, 0, 0, 0)),  # weight
            pl.BlockSpec((128,), lambda b, c, h, w: (0,)),  # conv_bias
            pl.BlockSpec((128,), lambda b, c, h, w: (0,)),  # bn_weight
            pl.BlockSpec((128,), lambda b, c, h, w: (0,)),  # bn_bias
        ),
        out_specs=pl.BlockSpec((8, 128, 126, 128), lambda b, c, h, w: (b, c, h, w)),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel", "parallel", "parallel")
        ),
    )(x, conv_weight, conv_bias, bn_weight, bn_bias)
