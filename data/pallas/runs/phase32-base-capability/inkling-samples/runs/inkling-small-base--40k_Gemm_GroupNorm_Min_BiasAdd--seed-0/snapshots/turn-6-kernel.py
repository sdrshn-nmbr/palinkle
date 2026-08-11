import jax
import jax.numpy as jnp
import jax.lax as lax
from jax import random
import jax.experimental.pallas as pl
import jax.experimental.pallas.tpu as pltpu

def workload(x, weight, linear_bias, gn_weight, gn_bias, bias):
    def kernel(x_ref, weight_ref, linear_bias_ref, gn_weight_ref, gn_bias_ref, bias_ref, out_ref):
        # Load inputs
        x_val = x_ref[...]
        w = weight_ref[...]
        lb = linear_bias_ref[...]
        gw = gn_weight_ref[...]
        gb = gn_bias_ref[...]
        b = bias_ref[...]
        
        # Matmul + linear bias
        # x: (4096, 8192), w: (8192, 8192) -> x @ w.T = (4096, 8192)
        # Actually weight is (8192, 8192), and we need x @ weight.T
        # But in JAX, jnp.dot(x, weight.T) or x @ weight.T
        # Since weight is square, weight.T = weight, but let's be precise
        y = jnp.dot(x_val, w.T) + lb
        
        # GroupNorm
        N, C = y.shape
        G = 512
        eps = 1e-5
        
        # Reshape to (N, G, C//G)
        y_reshaped = y.reshape(N, G, C // G)
        
        # Mean and var over axis=2
        mean = jnp.mean(y_reshaped, axis=2, keepdims=True)
        var = jnp.var(y_reshaped, axis=2, keepdims=True)
        
        # Normalize
        y_norm = (y_reshaped - mean) / jnp.sqrt(var + eps)
        
        # Reshape back
        y_norm = y_norm.reshape(N, C)
        
        # Scale and shift
        y_norm = y_norm * gw + gb
        
        # Min over axis=1 keepdims=True
        y_min = jnp.min(y_norm, axis=1, keepdims=True)
        
        # Reshape to (1, 1, N, 1)
        y_min = y_min.reshape(1, 1, N, 1)
        
        # Add bias (broadcasts to (1, 8192, 4096, 1))
        result = y_min + b
        
        out_ref[...] = result
    
    out_shape = jax.ShapeDtypeStruct((1, 8192, 4096, 1), x.dtype)
    
    return pl.pallas_call(
        kernel,
        out_shape=out_shape,
        grid=(),
        in_specs=(
            pl.no_block_spec,
            pl.no_block_spec,
            pl.no_block_spec,
            pl.no_block_spec,
            pl.no_block_spec,
            pl.no_block_spec,
        ),
        out_specs=pl.no_block_spec,
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel",)),
    )(x, weight, linear_bias, gn_weight, gn_bias, bias)
