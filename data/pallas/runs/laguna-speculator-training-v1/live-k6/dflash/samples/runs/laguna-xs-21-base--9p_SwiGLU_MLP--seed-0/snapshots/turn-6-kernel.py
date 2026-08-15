import jax
import jax.numpy as jnp
import jax.pallas as pl
import jax.pallas as plp
import jax.interpreters.tpu as pltpu

def workload(x, gate_kernel, up_kernel, down_kernel):
    """SwiGLU MLP kernel for Llama 3.1 70B.
    
    Computes: output = (SiLU(x @ gate) * (x @ up)) @ down
    """
    batch, seq_len, emb_dim = x.shape
    mlp_dim = gate_kernel.shape[1]
    
    # Block sizes for TPU - must be multiples of 8 for bf16
    block_m = 128  # tile along output feature dimension
    block_k = 128  # tile along reduction dimension
    
    def swiglu_mlp_kernel(ref_x, ref_gate_k, ref_up_k, ref_down_k, ref_out):
        # Grid indices
        b = pl.program_id(0)
        m = pl.program_id(1)
        n = pl.program_id(2)
        
        # Compute tile offsets
        x_offset_m = m * block_m
        x_offset_k = n * block_k
        
        # Accumulators for matmuls
        gate_accum = jnp.zeros((block_m, mlp_dim), dtype=jnp.float32)
        up_accum = jnp.zeros((block_m, mlp_dim), dtype=jnp.float32)
        
        # First matmul: x @ gate_kernel and x @ up_kernel
        for k in range(0, emb_dim, block_k):
            # Load x tile [block_m, block_k]
            x_tile = ref_x[
                b, 
                pl.index(x_offset_m, k, mode="clip"),
                pl.index(0, k, mode="clip")
            ][:block_m, :block_k]
            
            # Load gate kernel tile [block_k, mlp_dim]
            gate_k_tile = ref_gate_k[
                pl.index(k, 0, mode="clip"),
                pl.index(0, mlp_dim, mode="clip")
            ][:block_k, :mlp_dim]
            
            # Load up kernel tile [block_k, mlp_dim]
            up_k_tile = ref_up_k[
                pl.index(k, 0, mode="clip"),
                pl.index(0, mlp_dim, mode="clip")
            ][:block_k, :mlp_dim]
            
            # Convert to float32 for accumulation
            x_f32 = x_tile.astype(jnp.float32)
            gate_k_f32 = gate_k_tile.astype(jnp.float32)
            up_k_f32 = up_k_tile.astype(jnp.float32)
            
            # Accumulate
            gate_accum = gate_accum + jnp.dot(x_f32, gate_k_f32)
            up_accum = up_accum + jnp.dot(x_f32, up_k_f32)
        
        # Apply SiLU to gate and multiply with up
        gate_silu = jax.nn.silu(gate_accum.astype(jnp.bfloat16))
        gate_up = (gate_silu * up_accum.astype(jnp.bfloat16)).astype(jnp.float32)
        
        # Second matmul: (gate * up) @ down_kernel
        out_accum = jnp.zeros((block_m, emb_dim), dtype=jnp.float32)
        
        for k in range(0, mlp_dim, block_k):
            # Load gate_up tile [block_m, block_k]
            gate_up_tile = gate_up[
                pl.index(0, k, mode="clip"),
                pl.index(0, emb_dim, mode="clip")
            ][:block_m, :block_k]
            
            # Load down kernel tile [block_k, emb_dim]
            down_k_tile = ref_down_k[
                pl.index(k, 0, mode="clip"),
                pl.index(0, emb_dim, mode="clip")
            ][:block_k, :emb_dim]
            
            down_k_f32 = down_k_tile.astype(jnp.float32)
            
            out_accum = out_accum + jnp.dot(gate_up_tile, down_k_f32)
        
        # Write output
        ref_out[b, m, n] = out_accum.astype(jnp.bfloat16)
    
    # Grid dimensions
    grid_b = batch
    grid_m = (seq_len + block_m - 1) // block_m
    grid_n = (emb_dim + block_k - 1) // block_k
    
    # Define block specs
    x_spec = pl.BlockSpec(
        (batch, block_m, block_k),
        lambda b, m, n: (b, m * block_m, 0)
    )
    gate_k_spec = pl.BlockSpec(
        (block_k, mlp_dim),
        lambda b, m, n: (n * block_k, 0)
    )
    up_k_spec = pl.BlockSpec(
        (block_k, mlp_dim),
        lambda b, m, n: (n * block_k, 0)
    )
    down_k_spec = pl.BlockSpec(
        (mlp_dim, emb_dim),
        lambda b, m, n: (0, m * block_m)
    )
    out_spec = pl.BlockSpec(
        (batch, block_m, emb_dim),
        lambda b, m, n: (b, m * block_m, 0)
    )
    
    return pl.pallas_call(
        swiglu_mlp_kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        grid=(grid_b, grid_m, grid_n),
        in_specs=(x_spec, gate_k_spec, up_k_spec, down_k_spec),
        out_specs=out_spec,
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel", "parallel")
        ),
    )(x, gate_kernel, up_kernel, down_kernel)
