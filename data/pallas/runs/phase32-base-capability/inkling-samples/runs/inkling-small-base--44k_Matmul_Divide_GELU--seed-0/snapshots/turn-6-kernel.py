import jax
import jax.numpy as jnp
import jax.experimental.pallas as pl
import jax.experimental.pallas.tpu as pltpu

def workload(x, weight, bias):
    batch_size, input_size = x.shape
    output_size = weight.shape[1]
    
    # Block sizes
    block_m = 128
    block_n = 128
    block_k = 128
    
    def matmul_kernel(x_ref, w_ref, b_ref, out_ref):
        # Get program IDs
        pid_m = pl.program_id(0)
        pid_n = pl.program_id(1)
        
        # Initialize accumulator in float32 VMEM
        acc = jnp.zeros((block_m, block_n), dtype=jnp.float32)
        
        # Loop over reduction dimension
        for k in range(0, input_size, block_k):
            # Load x block: (block_m, block_k)
            x_block = pl.load(x_ref, (pl.dslice(pid_m * block_m, block_m), pl.dslice(k, block_k)))
            # Load weight block: (block_k, block_n)
            w_block = pl.load(w_ref, (pl.dslice(k, block_k), pl.dslice(pid_n * block_n, block_n)))
            
            # Convert to float32 for accumulation
            x_f32 = x_block.astype(jnp.float32)
            w_f32 = w_block.astype(jnp.float32)
            
            # Matmul accumulation
            acc += jnp.dot(x_f32, w_f32)
        
        # Add bias (broadcast over m dimension)
        # Load bias block for this n tile
        b_block = pl.load(b_ref, (pl.dslice(pid_n * block_n, block_n),))
        b_f32 = b_block.astype(jnp.float32)
        
        # Add bias: acc is (block_m, block_n), b is (block_n,)
        acc += b_f32[None, :]
        
        # Divide by 10.0
        acc = acc / 10.0
        
        # GELU
        # jax.nn.gelu approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        # But we can use jax.nn.gelu directly if available in kernel
        # Actually let's implement manually or use jnp
        # For simplicity, use jax.nn.gelu if it works in pallas
        # But to be safe, let's implement approximate gelu
        
        # Actually let's try using jax.nn.gelu
        out_f32 = jax.nn.gelu(acc)
        
        # Write back as bfloat16
        out_ref[...] = out_f32.astype(jnp.bfloat16)
    
    # Actually, using pl.load with dslice might not work exactly like that.
    # Let me use BlockSpec instead.
    pass
