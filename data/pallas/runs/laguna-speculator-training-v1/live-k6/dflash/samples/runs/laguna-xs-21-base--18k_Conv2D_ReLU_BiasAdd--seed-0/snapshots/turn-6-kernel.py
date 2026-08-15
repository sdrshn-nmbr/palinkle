import jax
import jax.numpy as jnp
import jax.pallas as pl
import jax.pallas as plp
from jax.experimental import enable_x64

def workload(x, weight, conv_bias, bias):
    """Conv2D + ReLU + BiasAdd kernel."""
    
    # Input shapes after transposes:
    # x: [128, 128, 128, 64] (NHWC)
    # weight: [3, 3, 64, 128] (HWIO)
    # Output: [128, 128, 126, 128] (NHWC before final transpose)
    
    batch_size = 128
    out_channels = 128
    out_height = 126
    out_width = 126
    kernel_h, kernel_w = 3, 3
    in_channels = 64
    
    def conv_kernel(x_ref, weight_ref, conv_bias_ref, bias_ref, out_ref):
        # Get program IDs for parallel dimensions
        b = pl.program_id(0)  # batch
        oc = pl.program_id(1)  # output channel
        oh = pl.program_id(2)  # output height
        ow = pl.program_id(3)  # output width
        
        # Accumulate convolution result in float32
        acc = 0.0
        
        # Iterate over kernel and input channels
        for kh in range(kernel_h):
            for kw in range(kernel_w):
                for ic in range(in_channels):
                    # Read input value
                    x_val = x_ref[b, oh, ow, ic].astype(jnp.float32)
                    # Read weight value
                    w_val = weight_ref[kh, kw, ic, oc].astype(jnp.float32)
                    # Accumulate
                    acc += x_val * w_val
        
        # Add conv_bias
        acc += conv_bias_ref[oc].astype(jnp.float32)
        
        # Apply ReLU
        acc = jnp.maximum(acc, 0.0)
        
        # Add bias
        acc += bias_ref[oc, 0, 0].astype(jnp.float32)
        
        # Convert back to bfloat16 and write output
        out_ref[b, oc, oh, ow] = acc.astype(jnp.bfloat16)
    
    # Grid dimensions: batch, out_channels, out_height, out_width
    grid = (batch_size, out_channels, out_height, out_width)
    
    # Block shapes for each dimension
    block_b = 1
    block_oc = 8  # Multiple of 8 for bf16
    block_oh = 8  # Multiple of 8 for bf16
    block_ow = 8  # Multiple of 8 for bf16
    
    # Define block specs
    x_spec = pl.BlockSpec((block_b, 128, block_oh, block_ow), 
                          lambda b, oc, oh, ow: (b, oh, ow, 0))
    weight_spec = pl.BlockSpec((kernel_h, kernel_w, in_channels, block_oc),
                               lambda b, oc, oh, ow: (0, 0, 0, oc))
    conv_bias_spec = pl.BlockSpec((block_oc,), lambda b, oc, oh, ow: (oc,))
    bias_spec = pl.BlockSpec((block_oc, 1, 1), lambda b, oc, oh, ow: (oc, 0, 0))
    out_spec = pl.BlockSpec((block_b, block_oc, block_oh, block_ow),
                            lambda b, oc, oh, ow: (b, oc, oh, ow))
    
    return pl.pallas_call(
        conv_kernel,
        out_shape=jax.ShapeDtypeStruct((batch_size, out_channels, out_height, out_width), jnp.bfloat16),
        grid=grid,
        in_specs=(x_spec, weight_spec, conv_bias_spec, bias_spec),
        out_specs=out_spec,
        compiler_params=plp.CompilerParams(
            dimension_semantics=("parallel", "parallel", "parallel", "parallel")
        ),
    )(x, weight, conv_bias, bias)
