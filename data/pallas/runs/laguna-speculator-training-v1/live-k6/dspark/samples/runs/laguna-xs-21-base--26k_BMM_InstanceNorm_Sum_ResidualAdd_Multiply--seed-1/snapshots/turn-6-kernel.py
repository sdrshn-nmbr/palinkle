import jax
import jax.numpy as jnp
import jax.pallas as pl
import jax.pallas as plp
import jax.interpreters.tpu as pltpu

def workload(x, y, bmm_weight, bmm_bias, in_weight, in_bias):
    """BMM + InstanceNorm + Sum + ResidualAdd + Multiply"""
    
    eps = 1e-5
    
    # BMM: x @ bmm_weight.T + bmm_bias
    x = jnp.dot(x, bmm_weight.T) + bmm_bias
    
    # Expand dims for instance norm: [batch, features] -> [batch, features, 1, 1]
    x = jnp.expand_dims(x, axis=2)  # [batch, features, 1]
    x = jnp.expand_dims(x, axis=3)  # [batch, features, 1, 1]
    
    # InstanceNorm: compute mean and variance over spatial dimensions (2, 3)
    mean = jnp.mean(x, axis=(2, 3), keepdims=True)
    var = jnp.var(x, axis=(2, 3), keepdims=True)
    
    # Normalize
    x = (x - mean) / jnp.sqrt(var + eps)
    
    # Reshape in_weight and in_bias for broadcasting
    # in_weight: [features] -> [1, 1, features, 1]
    # in_bias: [features] -> [1, 1, features, 1]
    in_weight_reshaped = jnp.reshape(in_weight, (1, 1, -1, 1))
    in_bias_reshaped = jnp.reshape(in_bias, (1, 1, -1, 1))
    
    # Scale and shift
    x = x * in_weight_reshaped + in_bias_reshaped
    
    # Squeeze back to 2D: [batch, features, 1, 1] -> [batch, features]
    x = jnp.squeeze(x, axis=3)  # [batch, features, 1]
    x = jnp.squeeze(x, axis=2)  # [batch, features]
    
    # Sum with y
    x = x + y
    
    # Multiply by y
    x = x * y
    
    return x
