import jax
import jax.numpy as jnp
import jax.pallas as pl
import jax.pallas as plp
import jax.interpreters.tpu as pltpu

def workload(x, weight, conv_bias, in_weight, in_bias):
    """Conv2d + InstanceNorm + Divide kernel."""
    
    # Input shapes:
    # x: [128, 64, 128, 128] NCHW
    # weight: [128, 64, 3, 3] OIHW
    # conv_bias: [128]
    # in_weight: [128]
    # in_bias: [128]
    # Output: [128, 128, 126, 126] NCHW
    
    batch_size = 128
    in_channels = 64
    out_channels = 128
    h, w = 128, 128
    kh, kw = 3, 3
    out_h, out_w = 126, 126
    
    # Block size for tiling
    block_batch = 16
    block_out_ch = 32
    block_h = 32
    block_w = 32
    
    def conv_inst_norm_kernel(
        x_ref, weight_ref, conv_bias_ref, in_weight_ref, in_bias_ref, out_ref
    ):
        # Get program IDs for tiling
        batch_idx = pl.program_id(0)
        out_ch_idx = pl.program_id(1)
        h_idx = pl.program_id(2)
        w_idx = pl.program_id(3)
        
        # Compute tile ranges
        batch_start = batch_idx * block_batch
        batch_end = min(batch_start + block_batch, batch_size)
        out_ch_start = out_ch_idx * block_out_ch
        out_ch_end = min(out_ch_start + block_out_ch, out_channels)
        h_start = h_idx * block_h
        h_end = min(h_start + block_h, out_h)
        w_start = w_idx * block_w
        w_end = min(w_start + block_w, out_w)
        
        # Initialize accumulator for convolution result
        # Shape: [batch_tile, out_ch_tile, h_tile, w_tile]
        acc = jnp.zeros(
            (batch_end - batch_start, out_ch_end - out_ch_start, 
             h_end - h_start, w_end - w_start), dtype=jnp.float32
        )
        
        # Perform convolution
        # Input is NCHW, kernel is OIHW
        # For each output position, we need to sum over input channels and kernel
        
        for in_ch in range(in_channels):
            for ky in range(kh):
                for kx in range(kw):
                    # Input position for this kernel element
                    in_h = h_start + ky
                    in_w = w_start + kx
                    
                    # Load input tile: [batch_tile, in_ch]
                    # Input shape is [batch, in_ch, h, w]
                    x_tile = x_ref[
                        batch_start:batch_end, in_ch, in_h:h_end, in_w:w_end
                    ]
                    
                    # Load kernel tile: [out_ch_tile]
                    # Kernel shape is [out_ch, in_ch, kh, kw]
                    k_tile = weight_ref[
                        out_ch_start:out_ch_end, in_ch, ky, kx
                    ]
                    
                    # Accumulate
                    acc = acc + x_tile.astype(jnp.float32) * k_tile.astype(jnp.float32)
        
        # Add bias
        bias_tile = conv_bias_ref[out_ch_start:out_ch_end]
        acc = acc + bias_tile.reshape(1, -1, 1, 1).astype(jnp.float32)
        
        # Instance normalization
        # Compute mean and variance over spatial dimensions
        mean = jnp.mean(acc, axis=(2, 3), keepdims=True)
        var = jnp.var(acc, axis=(2, 3), keepdims=True)
        
        # Normalize
        acc = (acc - mean) / jnp.sqrt(var + 1e-5)
        
        # Scale and shift
        scale_tile = in_weight_ref[out_ch_start:out_ch_end]
        shift_tile = in_bias_ref[out_ch_start:out_ch_end]
        acc = acc * scale_tile.reshape(1, -1, 1, 1).astype(jnp.float32)
        acc = acc + shift_tile.reshape(1, -1, 1, 1).astype(jnp.float32)
        
        # Divide by 2.0
        acc = acc / 2.0
        
        # Write output
        out_ref[
            batch_start:batch_end, out_ch_start:out_ch_end, 
            h_start:h_end, w_start:w_end
        ] = acc.astype(out_ref.dtype)
    
    # Calculate grid dimensions
    grid_batch = (batch_size + block_batch - 1) // block_batch
    grid_out_ch = (out_channels + block_out_ch - 1) // block_out_ch
    grid_h = (out_h + block_h - 1) // block_h
    grid_w = (out_w + block_w - 1) // block_w
    
    # Define block specs
    def make_block_spec(shape, ndim):
        return pl.BlockSpec(
            shape,
            lambda indices: tuple(s * b for s, b in zip(indices, shape))
        )
    
    # Output shape in NCHW
    out_shape = jax.ShapeDtypeStruct((batch_size, out_channels, out_h, out_w), x.dtype)
    
    return pl.pallas_call(
        conv_inst_norm_kernel,
        out_shape=out_shape,
        grid=(grid_batch, grid_out_ch, grid_h, grid_w),
        in_specs=(
            pl.BlockSpec((block_batch, in_channels, block_h + kh - 1, block_w + kw - 1), 
                        lambda batch, oc, h, w: (batch * block_batch, 0, h * block_h, w * block_w)),
            pl.BlockSpec((block_out_ch, in_channels, kh, kw),
                        lambda batch, oc, h, w: (oc * block_out_ch, 0, 0, 0)),
            pl.BlockSpec((block_out_ch,),
                        lambda batch, oc, h, w: (oc * block_out_ch,)),
            pl.BlockSpec((block_out_ch,),
                        lambda batch, oc, h, w: (oc * block_out_ch,)),
            pl.BlockSpec((block_out_ch,),
                        lambda batch, oc, h, w: (oc * block_out_ch,)),
        ),
        out_specs=pl.BlockSpec(
            (block_batch, block_out_ch, block_h, block_w),
            lambda batch, oc, h, w: (batch * block_batch, oc * block_out_ch, h * block_h, w * block_w)
        ),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel", "parallel", "parallel")
        ),
    )(x, weight, conv_bias, in_weight, in_bias)
