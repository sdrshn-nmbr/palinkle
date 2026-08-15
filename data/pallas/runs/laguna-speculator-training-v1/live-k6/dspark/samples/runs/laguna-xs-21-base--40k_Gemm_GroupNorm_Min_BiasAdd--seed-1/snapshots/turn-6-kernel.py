import jax
import jax.numpy as jnp
import jax.pallas as pl
import jax.pallas.lib as pllib
import jax.pallas.tpu as pltpu

def workload(x, weight, linear_bias, gn_weight, gn_bias, bias):
    """
    Fused kernel: Gemm + GroupNorm + Min + BiasAdd
    
    Operations:
    1. x = x @ weight + linear_bias
    2. Reshape to (N, G, C//G) where G=512
    3. GroupNorm: (x - mean) / sqrt(var + eps)
    4. Scale and shift: x * gn_weight + gn_bias
    5. Min along axis 1
    6. Reshape to (1, 1, N, 1)
    7. Add bias
    """
    num_groups = 512
    eps = 1e-5
    
    # Get shapes
    N, C = x.shape
    G = num_groups
    
    # Matmul with bias
    x = jnp.dot(x, weight, precision=jax.lax.Precision.DEFAULT)
    x = x + linear_bias
    
    # Reshape for group norm: (N, G, C//G)
    x = jnp.reshape(x, (N, G, C // G))
    
    # Group normalization
    mean = jnp.mean(x, axis=2, keepdims=True)
    var = jnp.var(x, axis=2, keepdims=True)
    x = (x - mean) / jnp.sqrt(var + eps)
    
    # Reshape back: (N, C)
    x = jnp.reshape(x, (N, C))
    
    # Scale and shift
    x = x * gn_weight + gn_bias
    
    # Min along axis 1
    x = jnp.min(x, axis=1, keepdims=True)
    
    # Reshape to (1, 1, N, 1)
    x = jnp.reshape(x, (1, 1, N, 1))
    
    # Add bias
    x = x + bias
    
    return x
