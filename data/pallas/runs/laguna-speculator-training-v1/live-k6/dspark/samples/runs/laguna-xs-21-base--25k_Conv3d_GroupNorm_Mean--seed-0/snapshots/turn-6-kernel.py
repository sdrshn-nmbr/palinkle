import jax
import jax.numpy as jnp
import jax.pallas as pl
import jax.pallas as plu
import jax.interpreters.tpu as pltpu

def workload(x, weight, conv_bias, gamma, beta):
    """Conv3D + GroupNorm + Mean workload."""
    
    # Configuration
    num_groups = 8
    
    # Step 1: Transpose x from [N, C, D, H, W] to [N, D, H, W, C] (NDHWC)
    x = jnp.transpose(x, (0, 2, 3, 4, 1))
    
    # Step 2: Transpose weight from [out_channels, kD, kH, kW, in_channels] to [kD, kH, kW, in_channels, out_channels] (DHWIO)
    kernel = jnp.transpose(weight, (2, 3, 4, 1, 0))
    
    # Step 3: Conv3D with VALID padding
    x = jax.lax.conv_general_dilated(
        x, kernel,
        window_strides=(1, 1, 1),
        padding='VALID',
        dimension_numbers=('NDHWC', 'DHWIO', 'NDHWC')
    )
    
    # Step 4: Add bias (reshaped to broadcast)
    x = x + jnp.reshape(conv_bias, (1, 1, 1, 1, -1))
    
    # Step 5: Transpose back to [N, C, D, H, W]
    x = jnp.transpose(x, (0, 4, 1, 2, 3))
    
    # Get shapes
    N, C, D, H, W = x.shape
    G = num_groups
    
    # Step 6: Reshape for GroupNorm: [N, G, C/G, D, H, W]
    x = jnp.reshape(x, (N, G, C // G, D, H, W))
    
    # Step 7: Compute mean over spatial dimensions (axes 2,3,4,5)
    mean = jnp.mean(x, axis=(2, 3, 4, 5), keepdims=True)
    
    # Step 8: Compute variance over spatial dimensions
    var = jnp.var(x, axis=(2, 3, 4, 5), keepdims=True)
    
    # Step 9: Normalize
    x = (x - mean) / jnp.sqrt(var + 1e-5)
    
    # Step 10: Reshape back to [N, C, D, H, W]
    x = jnp.reshape(x, (N, C, D, H, W))
    
    # Step 11: Scale and shift
    x = x * jnp.reshape(gamma, (1, C, 1, 1, 1)) + jnp.reshape(beta, (1, C, 1, 1, 1))
    
    # Step 12: Compute mean over axes (1, 2, 3, 4) to get [N]
    result = jnp.mean(x, axis=(1, 2, 3, 4))
    
    return result


def _conv3d_groupnorm_mean_kernel(x_ref, weight_ref, conv_bias_ref, gamma_ref, beta_ref, out_ref):
    """Pallas kernel for Conv3D + GroupNorm + Mean."""
    
    # Get grid indices
    n_idx = pl.program_id(0)
    
    # Load inputs for this batch element
    x = x_ref[n_idx]  # [3, 24, 32, 32]
    weight = weight_ref[:]  # [24, 3, 3, 3, 3]
    conv_bias = conv_bias_ref[:]  # [24]
    gamma = gamma_ref[:]  # [24]
    beta = beta_ref[:]  # [24]
    
    # Transpose x from [C, D, H, W] to [D, H, W, C] (NDHWC for single batch)
    x = jnp.transpose(x, (1, 2, 3, 0))
    
    # Transpose weight from [out_channels, kD, kH, kW, in_channels] to [kD, kH, kW, in_channels, out_channels]
    kernel = jnp.transpose(weight, (2, 3, 4, 1, 0))
    
    # Conv3D with VALID padding
    conv_out = jax.lax.conv_general_dilated(
        x[None, ...], kernel,  # Add batch dimension
        window_strides=(1, 1, 1),
        padding='VALID',
        dimension_numbers=('NDHWC', 'DHWIO', 'NDHWC')
    )
    conv_out = conv_out[0]  # Remove batch dimension: [D', H', W', out_channels]
    
    # Add bias
    conv_out = conv_out + conv_bias
    
    # Transpose to [out_channels, D', H', W']
    conv_out = jnp.transpose(conv_out, (3, 0, 1, 2))
    
    # GroupNorm parameters
    num_groups = 8
    G = num_groups
    C = 24
    D, H, W = conv_out.shape[1], conv_out.shape[2], conv_out.shape[3]
    
    # Reshape for GroupNorm: [G, C/G, D, H, W]
    x_reshaped = jnp.reshape(conv_out, (G, C // G, D, H, W))
    
    # Compute mean over spatial dimensions
    mean = jnp.mean(x_reshaped, axis=(2, 3, 4), keepdims=True)
    
    # Compute variance
    var = jnp.var(x_reshaped, axis=(2, 3, 4), keepdims=True)
    
    # Normalize
    normalized = (x_reshaped - mean) / jnp.sqrt(var + 1e-5)
    
    # Reshape back to [C, D, H, W]
    normalized = jnp.reshape(normalized, (C, D, H, W))
    
    # Scale and shift
    result = normalized * gamma[:, None, None, None] + beta[:, None, None, None]
    
    # Compute mean over axes (1, 2, 3) to get [C] then mean over C
    # Actually, we need mean over (1, 2, 3, 4) which gives scalar per batch
    # But the output is [128], so we need mean over all dimensions except batch
    final_mean = jnp.mean(result, axis=(0, 1, 2, 3))
    
    # Store output
    out_ref[n_idx] = final_mean


def workload(x, weight, conv_bias, gamma, beta):
    """Conv3D + GroupNorm + Mean workload using Pallas kernel."""
    
    batch_size = x.shape[0]
    
    return pl.pallas_call(
        _conv3d_groupnorm_mean_kernel,
        out_shape=jax.ShapeDtypeStruct((batch_size,), x.dtype),
        grid=(batch_size,),
        in_specs=(
            pl.BlockSpec((1,) + x.shape[1:], lambda i: (i,)),  # x: [1, C, D, H, W] per batch
            pl.BlockSpec(weight.shape, lambda i: tuple(0 for _ in weight.shape)),  # weight
            pl.BlockSpec(conv_bias.shape, lambda i: tuple(0 for _ in conv_bias.shape)),  # conv_bias
            pl.BlockSpec(gamma.shape, lambda i: tuple(0 for _ in gamma.shape)),  # gamma
            pl.BlockSpec(beta.shape, lambda i: tuple(0 for _ in beta.shape)),  # beta
        ),
        out_specs=pl.BlockSpec((batch_size,), lambda i: (i,)),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel",)
        ),
    )(x, weight, conv_bias, gamma, beta)
