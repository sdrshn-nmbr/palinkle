import jax
import jax.numpy as jnp
import jax.pallas as pl
import jax.pallas as plp
import jax.interpreters.tpu as pltpu

def workload(x, weight, bias, add_value):
    """
    Implements: Matmul + Add + Swish + Tanh + GELU + Hardtanh
    """
    batch_size, in_features = x.shape
    out_features = weight.shape[1]
    
    # Block size for TPU - must be multiple of 8 for bf16
    block_size = 128
    
    def kernel(ref_x, ref_weight, ref_bias, ref_add_value, ref_out):
        # Matmul: x @ weight + bias
        # Accumulate in float32 for better precision
        x_block = ref_x[...].astype(jnp.float32)
        weight_block = ref_weight[...].astype(jnp.float32)
        
        # Perform matrix multiplication
        # x_block: [batch_size, in_features]
        # weight_block: [in_features, out_features]
        # result: [batch_size, out_features]
        matmul_result = jnp.dot(x_block, weight_block)
        
        # Add bias
        bias_block = ref_bias[...].astype(jnp.float32)
        result = matmul_result + bias_block
        
        # Add add_value
        add_value_block = ref_add_value[...].astype(jnp.float32)
        result = result + add_value_block
        
        # Swish: x * sigmoid(x)
        result = result * jax.nn.sigmoid(result)
        
        # Tanh
        result = jnp.tanh(result)
        
        # GELU: x * sigmoid(x) (approximation)
        result = result * jax.nn.sigmoid(result)
        
        # Hardtanh: clip(x, -1, 1)
        result = jnp.clip(result, -1.0, 1.0)
        
        # Convert back to bfloat16
        ref_out[...] = result.astype(jnp.bfloat16)
    
    # Grid: one block per output element along the batch dimension
    # Each block processes a tile of the output
    grid = (batch_size // block_size,)
    
    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((batch_size, out_features), jnp.bfloat16),
        grid=grid,
        in_specs=(
            pl.BlockSpec((batch_size, in_features), lambda: (0, 0)),
            pl.BlockSpec((in_features, out_features), lambda: (0, 0)),
            pl.BlockSpec((out_features,), lambda: (0,)),
            pl.BlockSpec((out_features,), lambda: (0,)),
        ),
        out_specs=pl.BlockSpec((batch_size, out_features), lambda: (0, 0)),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel",)
        ),
    )(x, weight, bias, add_value)
