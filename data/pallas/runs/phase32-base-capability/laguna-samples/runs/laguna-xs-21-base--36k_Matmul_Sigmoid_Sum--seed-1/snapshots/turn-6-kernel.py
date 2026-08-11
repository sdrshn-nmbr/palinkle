import jax
import jax.numpy as jnp
import pallas as pl
import pytpu as pltpu

def matmul_sigmoid_sum_kernel(x_ref, weight_ref, bias_ref, out_ref):
    """Kernel that computes matmul + sigmoid + sum."""
    # Get block indices
    m = pl.program_id(0)  # batch dimension
    
    # Compute matmul for this row
    # x_ref: (8192,) - row of x
    # weight_ref: (8192, 8192) - full weight matrix
    # We need to compute x @ weight + bias
    
    # For each output element, compute the matmul
    # out_ref[m, 0] = sum(sigmoid(sum(x[m, k] * weight[k, j] for k) + bias[j]) for j)
    
    # Read the input row (vector)
    x_row = x_ref[:]  # Shape: (8192,)
    
    # Compute matmul: x_row @ weight_ref + bias_ref
    # This gives us intermediate values for all j
    matmul_result = jnp.dot(x_row, weight_ref) + bias_ref  # Shape: (8192,)
    
    # Apply sigmoid
    sigmoid_result = jax.nn.sigmoid(matmul_result)
    
    # Sum along the result
    sum_result = jnp.sum(sigmoid_result)
    
    # Write output
    out_ref[m, 0] = sum_result

def workload(x, weight, bias):
    """Compute matmul(x, weight) + bias, apply sigmoid, then sum along axis 1."""
    # Output shape
    out_shape = jax.ShapeDtypeStruct((x.shape[0], 1), x.dtype)
    
    # Grid: one program per batch element
    grid = (x.shape[0],)
    
    # Block sizes - use multiples of 8 for bfloat16
    x_block = (1, 8192)  # Full row of x for each batch element
    weight_block = (8192, 128)  # Process weight in chunks
    bias_block = (8192,)  # Full bias vector
    out_block = (1, 1)  # Single output element
    
    return pl.pallas_call(
        matmul_sigmoid_sum_kernel,
        out_shape=out_shape,
        grid=grid,
        in_specs=(
            pl.BlockSpec(x_block, lambda m: (m, slice(None))),
            pl.BlockSpec(weight_block, lambda m: (slice(None), slice(None))),
            pl.BlockSpec(bias_block, lambda m: slice(None)),
        ),
        out_specs=pl.BlockSpec(out_block, lambda m: (m, 0)),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel",)
        ),
    )(x, weight, bias)
