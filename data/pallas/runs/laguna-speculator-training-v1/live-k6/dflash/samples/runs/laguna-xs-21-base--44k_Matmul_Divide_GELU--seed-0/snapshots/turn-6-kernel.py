import jax
import jax.numpy as jnp
import jax.pallas as pl
import jax.pallas as plp
import jax.interpreters.tpu as pltpu


def workload(x, weight, bias):
    """Matmul + Divide + GELU kernel."""
    block_size = 128
    
    def kernel(ref_x, ref_weight, ref_bias, ref_out):
        # Matmul: x @ weight
        # Accumulate in float32 for better precision
        def matmul_kernel(x_ref, w_ref, out_ref):
            m, k = x_ref.shape
            n = w_ref.shape[1]
            
            # Initialize accumulator in float32
            acc = jnp.zeros((m, n), dtype=jnp.float32)
            
            # Tile along K dimension
            for i in range(k // block_size):
                x_block = x_ref[:, i*block_size:(i+1)*block_size].astype(jnp.float32)
                w_block = w_ref[i*block_size:(i+1)*block_size, :].astype(jnp.float32)
                acc = acc + jnp.dot(x_block, w_block)
            
            return acc
        
        # Compute matmul result
        matmul_result = matmul_kernel(ref_x, ref_weight, None)
        
        # Add bias (broadcast along first dimension)
        result = matmul_result + ref_bias.astype(jnp.float32)
        
        # Divide by 10.0
        result = result / 10.0
        
        # Apply GELU
        result = jax.nn.gelu(result)
        
        # Convert back to bfloat16 and write output
        ref_out[...] = result.astype(ref_out.dtype)
    
    # Grid: tile along output dimensions
    m, n = x.shape[0], weight.shape[1]
    grid = (m // block_size, n // block_size)
    
    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((m, n), x.dtype),
        grid=grid,
        in_specs=(
            pl.BlockSpec((block_size, block_size), lambda i, j: (i * block_size, slice(None))),
            pl.BlockSpec((block_size, block_size), lambda i, j: (slice(None), j * block_size)),
            pl.BlockSpec((block_size,), lambda i, j: (slice(None),)),
        ),
        out_specs=pl.BlockSpec((block_size, block_size), lambda i, j: (i * block_size, j * block_size)),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel")
        ),
    )(x, weight, bias)
