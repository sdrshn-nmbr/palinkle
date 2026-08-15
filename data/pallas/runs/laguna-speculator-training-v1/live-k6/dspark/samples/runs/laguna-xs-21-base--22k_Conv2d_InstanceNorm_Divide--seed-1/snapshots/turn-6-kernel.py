import jax
import jax.numpy as jnp
import jax.lax as lax
import pallas as pl
import pallas.triton as pltpu

def workload(x, weight, conv_bias, in_weight, in_bias):
    """Conv2d + InstanceNorm + Divide kernel."""
    
    # Output shape: [128, 128, 126, 126]
    out_shape = jax.ShapeDtypeStruct((128, 128, 126, 126), jnp.bfloat16)
    
    # Define block sizes for TPU efficiency
    # For bf16, use multiples of 8
    # For vectorized dimensions, use 128
    block_m = 128  # batch dimension
    block_n = 128  # output channels
    block_k = 64   # input channels
    
    def conv2d_instance_norm_kernel(ref_x, ref_weight, ref_conv_bias, 
                                     ref_in_weight, ref_in_bias, ref_out):
        """Pallas kernel for Conv2d + InstanceNorm + Divide."""
        
        # Get program IDs for tiling
        m_idx = pl.program_id(0)  # batch index
        n_idx = pl.program_id(1)  # output channel index
        
        # Load input x in NHWC format: [128, 128, 128, 64]
        # Original x is NCHW: [128, 64, 128, 128]
        # We need to transpose during loading
        
        # Load weight in HWIO format: [3, 3, 64, 128]
        # Original weight is OIH: [128, 64, 3, 3]
        # We need to transpose during loading
        
        # For simplicity, load entire slices and compute
        # This is a straightforward implementation
        
        # Load input patch for conv
        # x_nhwc shape: [128, 128, 128, 64]
        # We need to extract patches for the conv
        
        # For each output position (h, w), we need a 3x3 patch from input
        # Output spatial size: 126 x 126
        
        # Initialize output accumulator
        out = jnp.zeros((126, 126), dtype=jnp.float32)
        
        # Perform convolution using matrix multiplication
        # Reshape input to [128, 128*128, 64] and weight to [64*9, 128]
        # Then matmul: [128, 128*128, 128]
        
        # For each batch and output channel, compute conv
        for b in range(block_m):
            for oc in range(block_n):
                # Extract input patch for this batch
                # Input in NHWC: [128, 128, 128, 64]
                # We need to compute conv over spatial dimensions
                
                # Load the 3x3 kernel for this output channel
                # Weight in HWIO: [3, 3, 64, 128]
                kernel_slice = ref_weight[oc]  # [3, 3, 64]
                
                # Compute convolution
                result = jnp.zeros((126, 126), dtype=jnp.float32)
                
                for kh in range(3):
                    for kw in range(3):
                        # Load input slice
                        # Input in NHWC: [128, 128, 128, 64]
                        # For batch b, we need input[b, :, :, :]
                        input_slice = ref_x[b]  # [128, 128, 64]
                        
                        # Extract the patch at position (kh, kw)
                        # Shape: [126, 126, 64]
                        patch = input_slice[kh:kh+126, kw:kw+126, :]
                        
                        # Load kernel weights for this kernel position
                        # kernel_slice: [3, 3, 64]
                        k = kernel_slice[kh, kw, :]  # [64]
                        
                        # Element-wise multiply and accumulate
                        # patch: [126, 126, 64], k: [64]
                        result = result + jnp.sum(patch * k, axis=-1)
                
                # Add bias
                bias = ref_conv_bias[oc]
                result = result + bias
                
                # Store intermediate result for instance norm
                # We'll need to do instance norm across spatial dimensions
                # For now, store in a temporary
                
                # Transpose to NCHW for instance norm
                # result is [126, 126], need to add channel dim
                result_nchw = result  # [126, 126] - single channel for now
                
                # Compute mean and variance across spatial dimensions
                mean = jnp.mean(result_nchw)
                var = jnp.var(result_nchw)
                
                # Normalize
                normalized = (result_nchw - mean) / jnp.sqrt(var + 1e-5)
                
                # Scale and shift
                scale = ref_in_weight[oc]
                shift = ref_in_bias[oc]
                normalized = normalized * scale + shift
                
                # Divide by 2.0
                normalized = normalized / 2.0
                
                # Store output
                ref_out[b, oc] = normalized.astype(jnp.bfloat16)
    
    # Use a simpler approach - compute everything in the kernel
    # with proper tiling for TPU
    
    def kernel(ref_x, ref_weight, ref_conv_bias, ref_in_weight, ref_in_bias, ref_out):
        # Get program IDs
        m_idx = pl.program_id(0)
        n_idx = pl.program_id(1)
        
        # Load entire input and weight for this tile
        # x in NHWC: [128, 128, 128, 64]
        # weight in HWIO: [3, 3, 64, 128]
        
        # For simplicity, process one output element at a time
        # This is not optimal but correct
        
        # Get batch and channel indices
        b = m_idx
        oc = n_idx
        
        # Load input in NHWC format
        # x is NCHW: [128, 64, 128, 128]
        # We need to transpose to NHWC: [128, 128, 128, 64]
        
        # For each spatial position, compute conv
        out_h = 126
        out_w = 126
        
        # Initialize output
        result = jnp.zeros((out_h, out_w), dtype=jnp.float32)
        
        # Get kernel weights for this output channel
        # weight is OIH: [128, 64, 3, 3]
        # We need HWIO: [3, 3, 64, 128]
        # So for channel oc, kernel is [3, 3, 64]
        
        for kh in range(3):
            for kw in range(3):
                # Load input slice for this kernel position
                # x is NCHW: [128, 64, 128, 128]
                # For batch b, input is [64, 128, 128]
                # We need to extract [126, 126, 64] for conv
                
                # Input at position (kh, kw) in kernel
                # Shape: [126, 126, 64]
                input_patch = ref_x[b, :, kh:kh+126, kw:kw+126]  # [64, 126, 126]
                
                # Kernel weight for this position
                # weight is OIH: [128, 64, 3, 3]
                # For output channel oc, kernel is [64, 3, 3]
                # At position (kh, kw), we have [64]
                k = ref_weight[oc, :, kh, kw]  # [64]
                
                # Multiply and accumulate
                # input_patch: [64, 126, 126]
                # k: [64]
                # result: [126, 126]
                result = result + jnp.sum(input_patch * k[:, None, None], axis=0)
        
        # Add bias
        result = result + ref_conv_bias[oc]
        
        # Instance normalization
        # Compute mean and variance across spatial dimensions
        mean = jnp.mean(result)
        var = jnp.var(result)
        
        # Normalize
        result = (result - mean) / jnp.sqrt(var + 1e-5)
        
        # Scale and shift
        result = result * ref_in_weight[oc] + ref_in_bias[oc]
        
        # Divide by 2.0
        result = result / 2.0
        
        # Store output
        ref_out[b, oc] = result.astype(jnp.bfloat16)
    
    # Grid: 128 batches x 128 output channels
    grid = (128, 128)
    
    return pl.pallas_call(
        kernel,
        out_shape=out_shape,
        grid=grid,
        in_specs=(
            pl.BlockSpec((128, 64, 128, 128), lambda b, oc: (b, slice(None), slice(None), slice(None))),
            pl.BlockSpec((128, 64, 3, 3), lambda b, oc: (oc, slice(None), slice(None), slice(None))),
            pl.BlockSpec((128,), lambda b, oc: (oc,)),
            pl.BlockSpec((128,), lambda b, oc: (oc,)),
            pl.BlockSpec((128,), lambda b, oc: (oc,)),
        ),
        out_specs=pl.BlockSpec((128, 128), lambda b, oc: (b, oc)),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel")
        ),
    )(x, weight, conv_bias, in_weight, in_bias)
