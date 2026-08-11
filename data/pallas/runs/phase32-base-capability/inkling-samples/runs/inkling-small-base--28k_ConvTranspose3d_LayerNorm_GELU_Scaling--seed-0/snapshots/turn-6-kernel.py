import jax
import jax.numpy as jnp
import jax.lax as lax
from jax import nn
import jax.experimental.pallas as pl
import jax.experimental.pallas.tpu as pltpu

def workload(x, conv_weight, conv_bias, ln_weight, ln_bias):
    def kernel(x_ref, conv_weight_ref, conv_bias_ref, ln_weight_ref, ln_bias_ref, out_ref):
        x = x_ref[...]
        conv_weight = conv_weight_ref[...]
        conv_bias = conv_bias_ref[...]
        ln_weight = ln_weight_ref[...]
        ln_bias = ln_bias_ref[...]
        
        stride = 2
        padding = 1
        kernel_size = 4
        eps = 1e-05
        scaling_factor = 1.0
        
        # Transpose x: (0,2,3,4,1) -> (batch, d, h, w, channels) to (batch, channels, d, h, w)?
        # Wait, original: x.transpose((0,2,3,4,1))
        # x shape: (32, 32, 16, 32, 32) -> after transpose: (32, 16, 32, 32, 32)
        # Actually let's follow exactly.
        x = jnp.transpose(x, (0, 2, 3, 4, 1))
        
        # kernel = conv_weight.transpose((2,3,4,1,0))
        kernel = jnp.transpose(conv_weight, (2, 3, 4, 1, 0))
        # flip axes 0,1,2
        kernel = jnp.flip(kernel, axis=(0, 1, 2))
        
        batch_size, d_in, h_in, w_in, channels = x.shape
        k = kernel_size
        d_dilated = d_in + (d_in - 1) * (stride - 1)
        h_dilated = h_in + (h_in - 1) * (stride - 1)
        w_dilated = w_in + (w_in - 1) * (stride - 1)
        
        x_dilated = jnp.zeros((batch_size, d_dilated, h_dilated, w_dilated, channels), dtype=x.dtype)
        # Set with stride
        # x_dilated = x_dilated.at[:, ::stride, ::stride, ::stride, :].set(x)
        # But need to handle indexing properly
        slices = (slice(None), slice(None, None, stride), slice(None, None, stride), slice(None, None, stride), slice(None))
        x_dilated = x_dilated.at[slices].set(x)
        x = x_dilated
        
        pad = (k - 1) - padding
        jax_padding = ((pad, pad), (pad, pad), (pad, pad))
        
        # conv_general_dilated
        # dimension_numbers: NDHWC, DHWOI, NDHWC
        # window_strides = (1,1,1)
        x = lax.conv_general_dilated(
            x, kernel,
            window_strides=(1, 1, 1),
            padding=jax_padding,
            dimension_numbers=("NDHWC", "DHWOI", "NDHWC")
        )
        
        # Add bias: reshape conv_bias to (1,1,1,1,64) with -1 at end? 
        # Original: jnp.reshape(conv_bias, (1,1,1,1,-1))
        # Wait: reshape with (1,1,1,1,-1) -> (1,1,1,1,64)
        # But x after conv is (32, 32, 64, 64, 64)? Let's check.
        # Actually after transpose x is (32,16,32,32,32). After dilation: d_dilated = 16 + 15*1 = 31? Wait stride=2, so d_dilated = 16 + (16-1)*(2-1) = 31.
        # After conv with kernel 4, padding=pad, stride=1: output = (31 + 2*pad - 4)/1 + 1 = 31 + 2*2 - 4 + 1 = 32? Actually pad = 4-1-1 = 2. So 31+4-4+1 = 32.
        # So output is (32, 32, 32, 32, 64) in NDHWC? Wait dimension_numbers NDHWC means x is (N,D,H,W,C). After conv with DHWOI kernel, output is NDHWC.
        # So x shape = (32, 32, 32, 32, 64)? Wait d_in=16, h_in=32, w_in=32. After dilation: d=31, h=63, w=63. After conv: d=31+4-4+1=32, h=63+4-4+1=64, w=63+4-4+1=64. So (32,32,64,64,64).
        # Then transpose (0,4,1,2,3) -> (32,64,32,64,64). That matches output.
        
        x = x + jnp.reshape(conv_bias, (1, 1, 1, 1, -1))
        
        # Transpose: (0,4,1,2,3)
        x = jnp.transpose(x, (0, 4, 1, 2, 3))
        
        # LayerNorm on last axis (axis=-1)
        mean = jnp.mean(x, axis=-1, keepdims=True)
        var = jnp.mean((x - mean) ** 2, axis=-1, keepdims=True)
        x = (x - mean) / jnp.sqrt(var + eps)
        x = x * ln_weight + ln_bias
        
        # GELU
        x = nn.gelu(x)
        
        # Scaling
        x = x * scaling_factor
        
        out_ref[...] = x
    
    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((32, 64, 32, 64, 64), jnp.bfloat16),
        grid=(),
        in_specs=(
            pl.no_block_spec,
            pl.no_block_spec,
            pl.no_block_spec,
            pl.no_block_spec,
            pl.no_block_spec,
        ),
        out_specs=pl.no_block_spec,
    )(x, conv_weight, conv_bias, ln_weight, ln_bias)
