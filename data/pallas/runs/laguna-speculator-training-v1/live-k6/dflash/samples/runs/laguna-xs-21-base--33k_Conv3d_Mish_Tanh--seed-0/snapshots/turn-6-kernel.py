import jax
import jax.numpy as jnp
import jax.pallas as pl
import jax.pallas as pltpu


def workload(x, weight, bias):
    """Conv3D + Mish + Tanh kernel."""
    # Output shape after all operations
    out_shape = (16, 64, 30, 62, 62)
    
    def conv3d_mish_tanh_kernel(
        x_ref, weight_ref, bias_ref, out_ref
    ):
        # Get program IDs for output indices
        b = pl.program_id(0)  # batch
        oc = pl.program_id(1)  # output channel
        od = pl.program_id(2)  # output depth
        oh = pl.program_id(3)  # output height
        ow = pl.program_id(4)  # output width
        
        # Accumulate convolution result in float32 for numerical stability
        acc = 0.0
        
        # Loop over input channels and kernel dimensions
        for ic in range(32):
            for kd in range(3):
                for kh in range(3):
                    for kw in range(3):
                        # Input indices in NDHWC format
                        # x after transpose: [16, 32, 64, 64, 32]
                        # x[b, ic, od+kd, oh+kh, ow+kw]
                        x_val = x_ref[b, ic, od + kd, oh + kh, ow + kw]
                        # Weight indices in DHWIO format
                        # weight after transpose: [3, 3, 3, 32, 64]
                        # weight[kd, kh, kw, ic, oc]
                        w_val = weight_ref[kd, kh, kw, ic, oc]
                        acc += float(x_val) * float(w_val)
        
        # Add bias
        acc += float(bias_ref[oc])
        
        # Convert to bfloat16 for Mish computation
        x_val = jnp.bfloat16(acc)
        
        # Mish: x * tanh(ln(1 + exp(x)))
        # Compute ln(1 + exp(x)) = softplus(x)
        softplus_val = jnp.log(1.0 + jnp.exp(x_val))
        mish_val = x_val * jnp.tanh(softplus_val)
        
        # Apply tanh
        result = jnp.tanh(mish_val)
        
        out_ref[b, oc, od, oh, ow] = result
    
    # Grid dimensions match output shape
    grid = out_shape
    
    return pl.pallas_call(
        conv3d_mish_tanh_kernel,
        out_shape=jax.ShapeDtypeStruct(out_shape, jnp.bfloat16),
        grid=grid,
        in_specs=(
            # x_ref: [16, 32, 64, 64, 32] in NDHWC format
            pl.BlockSpec((16, 32, 64, 64, 32), lambda b, oc, od, oh, ow: (b, 0, 0, 0, 0)),
            # weight_ref: [3, 3, 3, 32, 64] in DHWIO format
            pl.BlockSpec((3, 3, 3, 32, 64), lambda b, oc, od, oh, ow: (0, 0, 0, 0, 0)),
            # bias_ref: [64]
            pl.BlockSpec((64,), lambda b, oc, od, oh, ow: (0,)),
        ),
        out_specs=pl.BlockSpec((1, 1, 1, 1, 1), lambda b, oc, od, oh, ow: (b, oc, od, oh, ow)),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel", "parallel", "parallel", "parallel")
        ),
    )(x, weight, bias)
