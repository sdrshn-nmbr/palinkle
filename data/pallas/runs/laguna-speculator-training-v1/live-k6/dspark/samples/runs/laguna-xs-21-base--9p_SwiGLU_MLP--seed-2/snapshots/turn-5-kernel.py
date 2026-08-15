import jax
import jax.numpy as jnp
import jax.pallas as pl
import jax.pallas as plp
import jax.interpreters.tpu as pltpu

def workload(x, gate_kernel, up_kernel, down_kernel):
    """SwiGLU MLP kernel for Llama 3.1 70B."""
    
    batch_size, seq_len, emb_dim = x.shape
    mlp_dim = gate_kernel.shape[1]
    
    # Define block sizes for TPU matmul
    # For bf16, block dimensions should be multiples of 8
    # Use 128-element tiling along vectorized dimensions
    block_m = 128  # tile along output dimension M
    block_k = 128  # tile along reduction dimension K
    block_n = 128  # tile along output dimension N
    
    # Grid for matmul: we need to cover all dimensions
    # x @ gate_kernel: [batch, seq_len, emb_dim] @ [emb_dim, mlp_dim] -> [batch, seq_len, mlp_dim]
    grid_m = (batch_size * seq_len + block_m - 1) // block_m
    grid_n = (mlp_dim + block_n - 1) // block_n
    
    def swiglu_kernel(ref_x, ref_gate_kernel, ref_up_kernel, ref_down_kernel, 
                      ref_gate_out, ref_up_out, ref_intermediate, ref_output):
        """Pallas kernel implementing SwiGLU MLP."""
        
        # Get program IDs for tiling
        m_idx = pl.program_id(0)
        n_idx = pl.program_id(1)
        k_idx = pl.program_id(2)
        
        # Compute tile offsets
        m_offset = m_idx * block_m
        n_offset = n_idx * block_n
        k_offset = k_idx * block_k
        
        # Initialize accumulators in float32 for better precision
        gate_acc = jnp.zeros((block_m, block_n), dtype=jnp.float32)
        up_acc = jnp.zeros((block_m, block_n), dtype=jnp.float32)
        
        # Perform the first matmuls: x @ gate_kernel and x @ up_kernel
        # We need to iterate over the reduction dimension (emb_dim)
        for k_tile in range((emb_dim + block_k - 1) // block_k):
            k_start = k_tile * block_k
            k_end = min(k_start + block_k, emb_dim)
            actual_k = k_end - k_start
            
            # Load x tile [block_m, actual_k]
            x_tile = ref_x[
                pl.broadcasted_index((m_offset + pl.arange(0, block_m), 
                                      pl.clamp(k_start, 0, emb_dim - actual_k)))
            ]
            
            # Load gate_kernel tile [actual_k, block_n]
            gate_k_tile = ref_gate_kernel[
                pl.clamp(k_start, 0, emb_dim - actual_k),
                pl.arange(0, block_n)
            ]
            
            # Load up_kernel tile [actual_k, block_n]
            up_k_tile = ref_up_kernel[
                pl.clamp(k_start, 0, emb_dim - actual_k),
                pl.arange(0, block_n)
            ]
            
            # Compute partial products and accumulate
            gate_acc = gate_acc + jnp.dot(x_tile, gate_k_tile).astype(jnp.float32)
            up_acc = up_acc + jnp.dot(x_tile, up_k_tile).astype(jnp.float32)
        
        # Apply SiLU activation to gate
        gate_out = jax.nn.silu(gate_acc).astype(x.dtype)
        
        # Store intermediate results
        ref_gate_out[m_idx * block_m + pl.arange(0, block_m), 
                     n_idx * block_n + pl.arange(0, block_n)] = gate_out
        
        ref_up_out[m_idx * block_m + pl.arange(0, block_m), 
                   n_idx * block_n + pl.arange(0, block_n)] = up_acc.astype(x.dtype)
        
        # Now compute (gate * up) @ down_kernel
        # gate * up is element-wise multiplication
        intermediate = gate_out * up_acc.astype(x.dtype)
        
        # Store intermediate for second matmul
        ref_intermediate[m_idx * block_m + pl.arange(0, block_m),
                         n_idx * block_n + pl.arange(0, block_n)] = intermediate
    
    # For simplicity, let's use a different approach with explicit matmuls
    # using jnp.dot inside the kernel
    
    def matmul_kernel(x_ref, gate_k_ref, up_k_ref, down_k_ref, out_ref):
        """Simplified kernel using jnp.dot for TPU matmul."""
        
        # Perform x @ gate_kernel
        gate = jnp.dot(x_ref[...], gate_k_ref[...])
        
        # Apply SiLU activation
        gate = jax.nn.silu(gate)
        
        # Perform x @ up_kernel
        up = jnp.dot(x_ref[...], up_k_ref[...])
        
        # Element-wise multiplication
        intermediate = gate * up
        
        # Perform final matmul
        output = jnp.dot(intermediate, down_k_ref[...])
        
        out_ref[...] = output
    
    # Use pallas_call with appropriate grid and specs
    # For TPU, we need to handle the 3D input x
    
    # Flatten batch and seq_len for simpler indexing
    flat_batch_seq = batch_size * seq_len
    
    # Grid dimensions
    grid = (flat_batch_seq, mlp_dim // block_n)
    
    def index_map(i):
        return (i, 0)
    
    return pl.pallas_call(
        matmul_kernel,
        out_shape=jax.ShapeDtypeStruct((batch_size, seq_len, emb_dim), x.dtype),
        grid=grid,
        in_specs=(
            pl.BlockSpec((block_m, emb_dim), lambda i: (i[0], 0)),
            pl.BlockSpec((emb_dim, mlp_dim), lambda i: (0, i[1])),
            pl.BlockSpec((emb_dim, mlp_dim), lambda i: (0, i[1])),
            pl.BlockSpec((mlp_dim, emb_dim), lambda i: (i[1], 0)),
        ),
        out_specs=pl.BlockSpec((block_m, emb_dim), lambda i: (i[0], 0)),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel")
        ),
    )(x, gate_kernel, up_kernel, down_kernel)
