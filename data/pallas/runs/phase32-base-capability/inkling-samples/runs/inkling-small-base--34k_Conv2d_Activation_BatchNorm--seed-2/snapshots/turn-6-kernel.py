import jax
import jax.numpy as jnp
import jax.lax as lax
import jax.nn as jnn
import jax.experimental.pallas as pl
import jax.experimental.pallas.tpu as pltpu

def workload(x, conv_weight, conv_bias, bn_weight, bn_bias):
    def kernel(x_ref, w_ref, b_ref, bn_w_ref, bn_b_ref, out_ref):
        x = x_ref[...]
        conv_weight = w_ref[...]
        conv_bias = b_ref[...]
        bn_weight = bn_w_ref[...]
        bn_bias = bn_b_ref[...]
        
        eps = 1e-05
        
        # Transpose x: (0,2,3,1) -> wait, instruction says (0,2,3,1) for x
        # Original: x = jnp.transpose(x, (0, 2, 3, 1))
        x = jnp.transpose(x, (0, 2, 3, 1))
        
        # Transpose weight: (2,3,1,0)
        weight = jnp.transpose(conv_weight, (2, 3, 1, 0))
        
        # Conv
        x = lax.conv_general_dilated(
            x, weight,
            window_strides=(1, 1),
            padding="VALID",
            dimension_numbers=("NHWC", "HWIO", "NHWC")
        )
        
        # Add bias reshaped to (1,1,1,128) with values [1,1,1,-1] applied to conv_bias
        # Wait: reshape(conv_bias, (1,1,1,-1)) then add? Actually instruction says:
        # x = x + reshape(conv_bias, (1,1,1,-1)) but with specific reshape values
        # Looking at AST: reshape(conv_bias, (1, 1, 1, -1)) with unary minus on 1 for last? No.
        # The reshape args are: 1, 1, 1, UnaryOp(USub, 1) = -1
        # So reshape(conv_bias, (1, 1, 1, -1))
        x = x + jnp.reshape(conv_bias, (1, 1, 1, -1))
        
        # Transpose x: (0, 3, 1, 2)
        x = jnp.transpose(x, (0, 3, 1, 2))
        
        # Mish activation: softplus(x) then tanh, multiply with x
        softplus_x = jnn.softplus(x)
        x = jnp.multiply(jnp.tanh(softplus_x), x)
        
        # BatchNorm mean over axes (0, 2, 3) keepdims=True
        mean = jnp.mean(x, axis=(0, 2, 3), keepdims=True)
        
        # Variance
        var = jnp.mean(jnp.power(x - mean, 2), axis=(0, 2, 3), keepdims=True)
        
        # Reshape bn_weight and bn_bias to (1, -1, 1, 1)
        # AST: reshape(bn_weight, (1, -1, 1, 1))
        w = jnp.reshape(bn_weight, (1, -1, 1, 1))
        b = jnp.reshape(bn_bias, (1, -1, 1, 1))
        
        # BatchNorm: (x - mean) / sqrt(var + eps) * w + b
        x = jnp.multiply(
            jnp.divide(x - mean, jnp.sqrt(var + eps)),
            w
        ) + b
        
        # Final transpose? Wait, instruction ends with return x.
        # But let's check the output shape: (64, 128, 126, 126)
        # After all operations, x should be in NCHW format? Let's trace.
        # Start: x is NHWC (64, 64, 128, 128)
        # After transpose (0,2,3,1): (64, 128, 128, 64) -> this is NCHW? Actually (N, C, H, W) = (64, 128, 128, 64)? No.
        # Wait: original x shape is [64, 64, 128, 128] which is N=64, H=64, W=128, C=128? Or N=64, C=64, H=128, W=128?
        # The contract says x shape [64, 64, 128, 128]. After transpose (0,2,3,1) -> (64, 128, 128, 64)
        # Then conv with weight transposed (2,3,1,0) from [128,64,3,3] -> (3,3,64,128)
        # Conv dimension_numbers NHWC, HWIO, NHWC: input is NHWC, weight is HWIO, output NHWC
        # Input to conv: (64, 128, 128, 64) -> N=64, H=128, W=128, C=64
        # Weight: (3,3,64,128) -> H=3, W=3, I=64, O=128
        # Output: (64, 126, 126, 128) -> N=64, H=126, W=126, C=128
        # Then add bias reshaped (1,1,1,-1) -> broadcasts to output
        # Then transpose (0,3,1,2) -> (64, 128, 126, 126) -> NCHW
        # Then mish on NCHW
        # Then mean over (0,2,3) -> over N, H, W -> keepdims -> (1, 128, 1, 1)
        # Then batchnorm with w,b reshaped (1,-1,1,1) -> (1, 128, 1, 1)
        # Final output is (64, 128, 126, 126) which matches contract.
        
        out_ref[...] = x
    
    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((64, 128, 126, 126), jnp.bfloat16),
        grid=(),
        in_specs=(
            pl.no_block_spec,
            pl.no_block_spec,
            pl.no_block_spec,
            pl.no_block_spec,
            pl.no_block_spec,
        ),
        out_specs=pl.no_block_spec,
    )(x, conv_weight, conv_bias, bn_weight, bn_bias)
