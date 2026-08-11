import jax
import jax.numpy as jnp
import jax.lax as lax
from jax.scipy.special import logsumexp
import jax.experimental.pallas as pl
import jax.experimental.pallas.tpu as pltpu

def workload(x, conv_weight, conv_bias, gn_weight, gn_bias):
    def kernel(x_ref, conv_weight_ref, conv_bias_ref, gn_weight_ref, gn_bias_ref, out_ref):
        x = x_ref[...]
        conv_weight = conv_weight_ref[...]
        conv_bias = conv_bias_ref[...]
        gn_weight = gn_weight_ref[...]
        gn_bias = gn_bias_ref[...]
        
        groups = 16
        eps = 1e-05
        
        # Transpose x to NHWC
        x_nhwc = jnp.transpose(x, (0, 2, 3, 1))
        
        # Transpose kernel to HWIO
        kernel = jnp.transpose(conv_weight, (2, 3, 1, 0))
        
        # Conv
        x_conv = lax.conv_general_dilated(
            x_nhwc, kernel,
            window_strides=(1, 1),
            padding='VALID',
            dimension_numbers=('NHWC', 'HWIO', 'NHWC')
        )
        
        # Add bias
        x_conv = x_conv + jnp.reshape(conv_bias, (1, 1, 1, -1))
        
        # Transpose back to NCHW
        x_conv = jnp.transpose(x_conv, (0, 3, 1, 2))
        
        # GroupNorm
        N, C, H, W = x_conv.shape
        x_reshaped = jnp.reshape(x_conv, (N, groups, C // groups, H, W))
        mean = jnp.mean(x_reshaped, axis=(2, 3, 4), keepdims=True)
        var = jnp.var(x_reshaped, axis=(2, 3, 4), keepdims=True)
        x_reshaped = (x_reshaped - mean) / jnp.sqrt(var + eps)
        x = jnp.reshape(x_reshaped, (N, C, H, W))
        
        # Apply GN weight/bias
        x_norm = x * jnp.reshape(gn_weight, (1, -1, 1, 1)) + jnp.reshape(gn_bias, (1, -1, 1, 1))
        
        # Tanh
        x_tanh = jnp.tanh(x_norm)
        
        # HardSwish
        x_hard_swish = x_tanh * jnp.minimum(jnp.maximum(x_tanh + 3, 0), 6) / 6
        
        # Residual add
        x_res = x_conv + x_hard_swish
        
        # LogSumExp over axis=1
        x_logsumexp = logsumexp(x_res, axis=1, keepdims=True)
        
        out_ref[...] = x_logsumexp
    
    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((128, 1, 126, 126), jnp.bfloat16),
        grid=(),
        in_specs=(
            pl.no_block_spec,
            pl.no_block_spec,
            pl.no_block_spec,
            pl.no_block_spec,
            pl.no_block_spec,
        ),
        out_specs=pl.no_block_spec,
    )(x, conv_weight, conv_bias, gn_weight, gn_bias)
