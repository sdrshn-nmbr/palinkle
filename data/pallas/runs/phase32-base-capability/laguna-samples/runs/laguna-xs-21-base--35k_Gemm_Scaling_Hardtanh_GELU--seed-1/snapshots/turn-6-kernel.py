import jax
import jax.numpy as jnp
import pallas as pl
import pallas.lib as pllib
import triton.tools.pallas as pltpu

def kernel(y_ref, x_ref, weight_ref, bias_ref, out_ref, *args):
    # Get grid dimensions
    m = x_ref.shape[0]
    n = weight_ref.shape[1]
    k = x_ref.shape[1]
    
    # Matmul: out = x @ weight + bias
    # Use float32 accumulation for better precision
    acc = jnp.zeros((n,), dtype=jnp.float32)
    
    for i in range(m):
        for j in range(n):
            acc = jnp.zeros((k,), dtype=jnp.float32)
            for l in range(k):
                acc = acc.at[l] = y_ref[i, l] * weight_ref[l, j]
            out_ref[i, j] = acc.sum() + bias_ref[j]

def workload(x, weight, bias):
    # x: [4096, 8192], weight: [8192, 8192], bias: [8192]
    # output: [4096, 8192]
    
    # Matmul: x @ weight
    # Result shape: [4096, 8192]
    # Add bias: broadcast along last dimension
    # Scale by 0.5
    # Clip to [-2, 2]
    # GELU: x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    
    def compute_kernel(ref_ref, y_ref, o_ref, *args):
        # ref_ref[0] is the reference output for interpretation
        # y_ref is the input x
        # o_ref is the output
        
        # Get block indices
        i = pl.program_id(0)  # batch dimension
        j = pl.program_id(1)  # output feature dimension
        
        # Block sizes
        block_m = ref_ref.shape[0]
        block_n = ref_ref.shape[1]
        
        # Matmul with accumulation in float32
        acc = jnp.zeros((block_m, block_n), dtype=jnp.float32)
        
        # Tile over the reduction dimension
        for k in range(0, y_ref.shape[1], 128):
            # Load blocks
            y_block = y_ref[i * block_m:(i + 1) * block_m, k:min(k + 128, y_ref.shape[1])]
            w_block = weight_ref[k:min(k + 128, weight_ref.shape[0]), j * block_n:(j + 1) * block_n]
            
            # Compute partial result
            acc = acc.at[:, :].add(jnp.dot(y_block, w_block).astype(jnp.float32))
        
        # Add bias
        acc = acc + bias_ref[j * block_n:(j + 1) * block_n]
        
        # Scale by 0.5
        acc = acc * 0.5
        
        # Clip to [-2, 2]
        acc = jnp.clip(acc, -2.0, 2.0)
        
        # GELU
        acc = acc * 0.5 * (1.0 + jnp.tanh(jnp.sqrt(2.0 / jnp.pi) * (acc + 0.044715 * jnp.power(acc, 3.0))))
        
        # Store result
        o_ref[:] = acc.astype(ref_ref.dtype)
    
    # Define grid based on output shape
    grid = (x.shape[0] // 128, x.shape[1] // 128)
    
    return pl.pallas_call(
        compute_kernel,
        out_shape=jax.ShapeDtypeStruct((x.shape[0], weight.shape[1]), x.dtype),
        grid=grid,
        in_specs=(
            pl.BlockSpec((128, 128), lambda idx: (idx[0], idx[1])),
            pl.BlockSpec((x.shape[0], x.shape[1]), lambda idx: (idx[0] * 128, 0)),
            pl.BlockSpec((weight.shape[0], weight.shape[1]), lambda idx: (0, idx[1] * 128)),
        ),
        out_specs=pl.BlockSpec((128, 128), lambda idx: (idx[0], idx[1])),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel", "parallel")),
    )(x, weight, bias)
