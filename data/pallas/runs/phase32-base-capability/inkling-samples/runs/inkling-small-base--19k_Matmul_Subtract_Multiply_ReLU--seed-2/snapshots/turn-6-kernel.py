import jax
import jax.numpy as jnp
import jax.experimental.pallas as pl
import jax.experimental.pallas.tpu as pltpu

def workload(x, weight, bias):
    batch_size, in_features = x.shape
    out_features = weight.shape[1]
    
    # Tile sizes - multiples of 128 for TPU
    block_m = 128
    block_n = 128
    block_k = 128
    
    grid_m = (batch_size + block_m - 1) // block_m
    grid_n = (out_features + block_n - 1) // block_n
    
    def matmul_kernel(x_ref, w_ref, b_ref, out_ref):
        # Get program IDs
        pid_m = pl.program_id(0)
        pid_n = pl.program_id(1)
        
        # Initialize accumulator in float32 VMEM
        acc = jnp.zeros((block_m, block_n), dtype=jnp.float32)
        
        # Loop over k dimension
        num_k = (in_features + block_k - 1) // block_k
        for k in range(num_k):
            # Load x tile: (block_m, block_k)
            x_tile = pl.load(x_ref, (pl.dslice(pid_m * block_m, block_m), pl.dslice(k * block_k, block_k)))
            # Load w tile: (block_k, block_n)
            w_tile = pl.load(w_ref, (pl.dslice(k * block_k, block_k), pl.dslice(pid_n * block_n, block_n)))
            
            # Accumulate matmul in float32
            acc += jnp.dot(x_tile.astype(jnp.float32), w_tile.astype(jnp.float32))
        
        # Load bias tile
        b_tile = pl.load(b_ref, (pl.dslice(pid_n * block_n, block_n),))
        
        # Add bias
        result = acc + b_tile.astype(jnp.float32)
        
        # Subtract 2.0
        result = result - 2.0
        
        # Multiply 1.5
        result = result * 1.5
        
        # ReLU
        result = jnp.maximum(result, 0.0)
        
        # Store result as bfloat16
        pl.store(out_ref, (pl.dslice(pid_m * block_m, block_m), pl.dslice(pid_n * block_n, block_n)), result.astype(jnp.bfloat16))
    
    return pl.pallas_call(
        matmul_kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        grid=(grid_m, grid_n),
        in_specs=(
            pl.BlockSpec((block_m, block_k), lambda i, j: (i * block_m, 0)),
            pl.BlockSpec((block_k, block_n), lambda i, j: (0, j * block_n)),
            pl.BlockSpec((block_n,), lambda i, j: (j * block_n,)),
        ),
        out_specs=pl.BlockSpec((block_m, block_n), lambda i, j: (i * block_m, j * block_n)),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel", "parallel")),
    )(x, weight, bias)
