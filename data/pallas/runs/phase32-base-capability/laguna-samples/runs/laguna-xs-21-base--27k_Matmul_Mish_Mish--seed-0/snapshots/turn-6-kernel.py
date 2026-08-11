import jax
import jax.numpy as jnp
import pallas as pl
import pallas.lib as pllib
import jax.experimental.pallas as pl
from jax.experimental import pallas as pl
from jax.experimental.pallas import TPUCompilerParams
import jax.interpreters.pallas as pl

def matmul_mish_mish_kernel(
    x_ref,
    weight_ref,
    bias_ref,
    out_ref,
):
    """Kernel that performs matmul + mish + mish."""
    # Get grid dimensions
    m = pl.program_id(0)  # batch dimension (4096)
    n = pl.program_id(1)  # output dimension (8192)
    
    # Accumulate in float32 for better precision
    acc = 0.0
    
    # Matmul: x[m, k] @ weight[k, n] + bias[n]
    for k in range(x_ref.shape[1]):
        acc += x_ref[m, k] * weight_ref[k, n]
    
    # Add bias
    acc += bias_ref[n]
    
    # Convert to bfloat16 for Mish computation
    acc_bf16 = acc.astype(jnp.bfloat16)
    
    # First Mish: x * tanh(softplus(x))
    mish1 = acc_bf16 * jnp.tanh(jnp.nn.softplus(acc_bf16))
    
    # Second Mish: x * tanh(softplus(x))
    out = mish1 * jnp.tanh(jnp.nn.softplus(mish1))
    
    out_ref[m, n] = out


def workload(x, weight, bias):
    """Workload that performs matmul + mish + mish."""
    # Output shape
    out_shape = jax.ShapeDtypeStruct(x.shape, x.dtype)
    
    # Grid: (batch_size // block_m, output_size // block_n)
    # For TPU, use block sizes that are multiples of 8 for bf16
    block_m = 128
    block_n = 128
    
    grid = (x.shape[0] // block_m, x.shape[1] // block_n)
    
    return pl.pallas_call(
        matmul_mish_mish_kernel,
        out_shape=out_shape,
        grid=grid,
        in_specs=(
            pl.BlockSpec((block_m, x.shape[1]), lambda i: (i,)),  # x
            pl.BlockSpec((x.shape[0], block_n), lambda i: (i,)),  # weight
            pl.BlockSpec((block_n,), lambda i: (i,)),  # bias
        ),
        out_specs=pl.BlockSpec((block_m, block_n), lambda i: (i,)),
        compiler_params=TPUCompilerParams(
            dimension_semantics=("parallel", "parallel")
        ),
    )(x, weight, bias)
