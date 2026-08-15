import jax
import jax.numpy as jnp
import jax.pallas as pl
import jax.pallas as plp
import jax.interpreters.pallas as pallas
from jax.interpreters import pallas as pl
import jax.numpy as jnp

def workload(x, weight, conv_bias, bias):
    """Conv2D + ReLU + BiasAdd kernel."""
    
    # Input shapes:
    # x: [128, 64, 128, 128] (N, C, H, W) - NCHW format
    # weight: [128, 64, 3, 3] (out_ch, in_ch, kH, kW) - OIHW format
    # conv_bias: [128] - per output channel bias
    # bias: [128, 1, 1] - broadcastable bias
    
    # Output shape: [128, 128, 126, 126] (N, C, H, W) - NCHW format
    
    batch_size = 128
    in_channels = 64
    out_channels = 128
    kernel_size = 3
    input_h = 128
    input_w = 128
    output_h = 126
    output_w = 126
    
    def conv_relu_bias_kernel(
        x_ref,
        weight_ref,
        conv_bias_ref,
        bias_ref,
        out_ref,
    ):
        # Get program IDs for parallel dimensions
        n = pl.program_id(0)  # batch dimension
        oc = pl.program_id(1)  # output channel
        oh = pl.program_id(2)  # output height
        ow = pl.program_id(3)  # output width
        
        # Accumulator for convolution result
        acc = 0.0
        
        # Perform convolution over input channels and kernel
        for ic in range(in_channels):
            for kh in range(kernel_size):
                for kw in range(kernel_size):
                    # Input position
                    ih = oh + kh
                    iw = ow + kw
                    
                    # Load input value
                    x_val = x_ref[n, ic, ih, iw].astype(jnp.float32)
                    
                    # Load weight value
                    w_val = weight_ref[oc, ic, kh, kw].astype(jnp.float32)
                    
                    # Accumulate
                    acc += x_val * w_val
        
        # Add conv_bias
        acc += conv_bias_ref[oc].astype(jnp.float32)
        
        # Apply ReLU
        acc = jnp.maximum(acc, 0.0)
        
        # Add bias
        bias_val = bias_ref[oc, 0, 0].astype(jnp.float32)
        acc += bias_val
        
        # Store result (convert back to bfloat16)
        out_ref[n, oc, oh, ow] = acc.astype(jnp.bfloat16)
    
    # Grid dimensions
    grid = (batch_size, out_channels, output_h, output_w)
    
    # Define input specs
    x_spec = pl.BlockSpec(
        (batch_size, in_channels, input_h, input_w),
        lambda n, oc, oh, ow: (n, 0, 0, 0)
    )
    
    weight_spec = pl.BlockSpec(
        (out_channels, in_channels, kernel_size, kernel_size),
        lambda n, oc, oh, ow: (0, 0, 0, 0)
    )
    
    conv_bias_spec = pl.BlockSpec(
        (out_channels,),
        lambda n, oc, oh, ow: (oc,)
    )
    
    bias_spec = pl.BlockSpec(
        (out_channels, 1, 1),
        lambda n, oc, oh, ow: (oc, 0, 0)
    )
    
    out_spec = pl.BlockSpec(
        (batch_size, out_channels, output_h, output_w),
        lambda n, oc, oh, ow: (n, oc, oh, ow)
    )
    
    # Call the Pallas kernel
    return pl.pallas_call(
        conv_relu_bias_kernel,
        out_shape=jax.ShapeDtypeStruct((batch_size, out_channels, output_h, output_w), jnp.bfloat16),
        grid=grid,
        in_specs=(x_spec, weight_spec, conv_bias_spec, bias_spec),
        out_specs=out_spec,
        compiler_params=jax.pallas.tpu.CompilerParams(
            dimension_semantics=("parallel", "parallel", "parallel", "parallel")
        ),
    )(x, weight, conv_bias, bias)
