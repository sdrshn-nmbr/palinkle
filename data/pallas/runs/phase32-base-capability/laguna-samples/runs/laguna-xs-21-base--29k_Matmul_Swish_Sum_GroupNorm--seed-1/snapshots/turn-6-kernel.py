import jax
import jax.numpy as jnp
import pallas as pl
import jax.pallas as plp
import jaxlib.xla_extension as pltpu

# Constants from the workload
BATCH_SIZE = 8192
IN_FEATURES = 4096
OUT_FEATURES = 4096
NUM_GROUPS = 64
GROUP_SIZE = OUT_FEATURES // NUM_GROUPS  # 64

def workload(x, weight, bias, gn_weight, gn_bias):
    """
    Fused kernel: Matmul -> Swish -> Add Bias -> GroupNorm -> Scale and Shift
    
    The computation follows the canonical AST:
    1. x = x @ weight
    2. x = x * sigmoid(x)  (Swish)
    3. x = x + bias
    4. x = reshape(x, [-1, num_groups, group_size])
    5. mean = mean(x, axis=-1, keepdims=True)
    6. var = var(x, axis=-1, keepdims=True)
    7. x = (x - mean) / sqrt(var + 1e-5)
    8. x = reshape(x, [batch_size, out_features])
    9. x = x * gn_weight + gn_bias
    """
    
    # Step 1: Compute matmul
    # x: [BATCH_SIZE, IN_FEATURES], weight: [IN_FEATURES, OUT_FEATURES]
    # result: [BATCH_SIZE, OUT_FEATURES]
    matmul_result = jnp.matmul(x, weight)
    
    # Step 2: Apply Swish: x * sigmoid(x)
    # Note: The AST shows x * sigmoid(x) applied to the matmul result
    swish_result = matmul_result * jax.nn.sigmoid(matmul_result)
    
    # Step 3: Add bias
    # bias: [OUT_FEATURES]
    biased_result = swish_result + bias
    
    # Step 4: Reshape for group norm
    # Reshape to [batch_size, num_groups, group_size]
    reshaped = biased_result.reshape(BATCH_SIZE, NUM_GROUPS, GROUP_SIZE)
    
    # Step 5: Compute mean and variance along the last axis (group dimension)
    mean = jnp.mean(reshaped, axis=-1, keepdims=True)
    var = jnp.var(reshaped, axis=-1, keepdims=True)
    
    # Step 6: Normalize
    normalized = (reshaped - mean) / jnp.sqrt(var + 1e-5)
    
    # Step 7: Reshape back to [batch_size, out_features]
    normalized_flat = normalized.reshape(BATCH_SIZE, OUT_FEATURES)
    
    # Step 8: Apply scale and shift with group parameters
    # Reshape gn_weight and gn_bias to [num_groups, group_size] for proper broadcasting
    # Actually, looking at the AST, gn_weight and gn_bias are [out_features]
    # And they're applied element-wise after the reshape
    result = normalized_flat * gn_weight + gn_bias
    
    # Convert to bfloat16 as required
    return result.astype(jnp.bfloat16)
