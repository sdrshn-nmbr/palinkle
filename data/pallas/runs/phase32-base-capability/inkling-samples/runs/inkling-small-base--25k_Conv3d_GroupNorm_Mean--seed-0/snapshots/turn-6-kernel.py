import jax
import jax.numpy as jnp
import jax.lax as lax
import jax.experimental.pallas as pl
import jax.experimental.pallas.tpu as pltpu

def workload(x, weight, conv_bias, gamma, beta):
    def kernel(x_ref, weight_ref, conv_bias_ref, gamma_ref, beta_ref, out_ref):
        # Read full arrays
        x = x_ref[...]
        weight = weight_ref[...]
        conv_bias = conv_bias_ref[...]
        gamma = gamma_ref[...]
        beta = beta_ref[...]
        
        num_groups = 8
        
        # Transpose x: (0,2,3,4,1) -> (N, D, H, W, C) wait let's follow AST exactly
        # AST: jnp.transpose(x, (0, 2, 3, 4, 1))
        x = jnp.transpose(x, (0, 2, 3, 4, 1))
        
        # Transpose weight: (2,3,4,1,0)
        kernel_t = jnp.transpose(weight, (2, 3, 4, 1, 0))
        
        # conv_general_dilated
        x = lax.conv_general_dilated(
            x, kernel_t,
            window_strides=(1, 1, 1),
            padding="VALID",
            dimension_numbers=("NDHWC", "DHWIO", "NDHWC")
        )
        
        # Add bias reshaped to (1,1,1,1,24) with last dim -1? Wait AST: reshape(conv_bias, (1,1,1,1,-1))
        # Actually reshape with (1,1,1,1,-1) means last dim inferred = 24
        bias_reshaped = jnp.reshape(conv_bias, (1, 1, 1, 1, -1))
        x = x + bias_reshaped
        
        # Transpose x: (0,4,1,2,3)
        x = jnp.transpose(x, (0, 4, 1, 2, 3))
        
        # Get shape
        N, C, D, H, W = x.shape
        G = num_groups
        
        # Reshape to (N, G, C//G, D, H, W)
        x = jnp.reshape(x, (N, G, C // G, D, H, W))
        
        # Mean over axes 2,3,4,5 keepdims=True
        mean = jnp.mean(x, axis=(2, 3, 4, 5), keepdims=True)
        
        # Var over axes 2,3,4,5 keepdims=True
        var = jnp.var(x, axis=(2, 3, 4, 5), keepdims=True)
        
        # Normalize
        x = (x - mean) / jnp.sqrt(var + 1e-05)
        
        # Reshape back to (N, C, D, H, W)
        x = jnp.reshape(x, (N, C, D, H, W))
        
        # Apply gamma and beta
        # gamma reshaped to (1, -1, 1, 1, 1) -> (1, 24, 1, 1, 1)
        gamma_r = jnp.reshape(gamma, (1, -1, 1, 1, 1))
        beta_r = jnp.reshape(beta, (1, -1, 1, 1, 1))
        x = x * gamma_r + beta_r
        
        # Mean over axes 1,2,3,4 -> output (N,)
        out = jnp.mean(x, axis=(1, 2, 3, 4))
        
        out_ref[...] = out
    
    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((x.shape[0],), x.dtype),
        grid=(),
        in_specs=(
            pl.no_block_spec,
            pl.no_block_spec,
            pl.no_block_spec,
            pl.no_block_spec,
            pl.no_block_spec,
        ),
        out_specs=pl.no_block_spec,
    )(x, weight, conv_bias, gamma, beta)
