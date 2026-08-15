import jax
import jax.numpy as jnp
import jax.pallas as pl
import jax.pallas as plp
import jax.interpreters.tpu as pltpu

def workload(x, y, bmm_weight, bmm_bias, in_weight, in_bias):
    """BMM + InstanceNorm + Sum + ResidualAdd + Multiply"""
    
    # Constants
    eps = 1e-5
    
    # BMM: x @ bmm_weight.T + bmm_bias
    # x: [4096, 8192], bmm_weight: [8192, 8192], bmm_bias: [8192]
    # Result: [4096, 8192]
    x = jnp.dot(x, bmm_weight.T) + bmm_bias
    
    # Expand dims for instance norm: [4096, 8192] -> [4096, 8192, 1, 1]
    x = jnp.expand_dims(x, axis=2)
    x = jnp.expand_dims(x, axis=3)
    
    # InstanceNorm: compute mean and variance over axes (2, 3)
    mean = jnp.mean(x, axis=(2, 3), keepdims=True)
    var = jnp.var(x, axis=(2, 3), keepdims=True)
    
    # Normalize: (x - mean) / sqrt(var + eps)
    x = (x - mean) / jnp.sqrt(var + eps)
    
    # Scale and shift: reshape in_weight and in_bias to [1, 1, 1, 8192]
    in_weight_reshaped = jnp.reshape(in_weight, (1, 1, 1, 8192))
    in_bias_reshaped = jnp.reshape(in_bias, (1, 1, 1, 8192))
    x = x * in_weight_reshaped + in_bias_reshaped
    
    # Squeeze back: [4096, 8192, 1, 1] -> [4096, 8192]
    x = jnp.squeeze(x, axis=3)
    x = jnp.squeeze(x, axis=2)
    
    # Residual add and multiply: x + y, then * y
    x = x + y
    x = x * y
    
    return x
