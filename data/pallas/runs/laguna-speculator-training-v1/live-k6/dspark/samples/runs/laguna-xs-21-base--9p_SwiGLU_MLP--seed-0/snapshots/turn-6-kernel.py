import jax
import jax.numpy as jnp
import jax.pallas as pl
import jax.pallas as plp
import jax.interpreters.tpu as pltpu


def workload(x, gate_kernel, up_kernel, down_kernel):
    """SwiGLU MLP kernel for Llama 3.1 70B.
    
    Computes: output = (SiLU(x @ gate_kernel) * (x @ up_kernel)) @ down_kernel
    """
    batch, seq_len, emb_dim = x.shape
    mlp_dim = gate_kernel.shape[1]
    
    # Block size for matmul - use 128 for vectorized dimensions
    block_m = 128
    block_k = 128
    block_n = 128
    
    def swiglu_mlp_kernel(x_ref, gate_kernel_ref, up_kernel_ref, down_kernel_ref, out_ref):
        # Grid indices
        b = pl.program_id(0)
        m = pl.program_id(1)
        n = pl.program_id(2)
        
        # Compute gate = silu(x @ gate_kernel)
        # x[b, m, :] @ gate_kernel -> shape [mlp_dim]
        gate_result = jnp.zeros((mlp_dim,), dtype=jnp.float32)
        for k in range(0, emb_dim, block_k):
            x_block = x_ref[b, m, k:min(k + block_k, emb_dim)]
            gate_kernel_block = gate_kernel_ref[k:min(k + block_k, emb_dim), :]
            gate_result = gate_result + jnp.dot(x_block.astype(jnp.float32), 
                                                 gate_kernel_block.astype(jnp.float32))
        
        gate_result = jax.nn.silu(gate_result.astype(jnp.bfloat16))
        
        # Compute up = x @ up_kernel
        up_result = jnp.zeros((mlp_dim,), dtype=jnp.float32)
        for k in range(0, emb_dim, block_k):
            x_block = x_ref[b, m, k:min(k + block_k, emb_dim)]
            up_kernel_block = up_kernel_ref[k:min(k + block_k, emb_dim), :]
            up_result = up_result + jnp.dot(x_block.astype(jnp.float32),
                                             up_kernel_block.astype(jnp.float32))
        
        # Compute intermediate = gate * up
        intermediate = gate_result * up_result
        
        # Compute output = intermediate @ down_kernel
        out_result = jnp.zeros((emb_dim,), dtype=jnp.float32)
        for k in range(0, mlp_dim, block_k):
            intermediate_block = intermediate[k:min(k + block_k, mlp_dim)]
            down_kernel_block = down_kernel_ref[k:min(k + block_k, mlp_dim), :]
            out_result = out_result + jnp.dot(intermediate_block,
                                               down_kernel_block.astype(jnp.float32))
        
        out_ref[b, m, n] = out_result[n].astype(jnp.bfloat16)
    
    # Grid dimensions
    grid_b = batch
    grid_m = seq_len
    grid_n = emb_dim // block_n
    
    return pl.pallas_call(
        swiglu_mlp_kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        grid=(grid_b, grid_m, grid_n),
        in_specs=(
            pl.BlockSpec((batch, seq_len, emb_dim), lambda b, m, n: (b, m, n)),
            pl.BlockSpec((emb_dim, mlp_dim), lambda b, m, n: (0, 0)),
            pl.BlockSpec((emb_dim, mlp_dim), lambda b, m, n: (0, 0)),
            pl.BlockSpec((mlp_dim, emb_dim), lambda b, m, n: (0, 0)),
        ),
        out_specs=pl.BlockSpec((batch, seq_len, emb_dim), lambda b, m, n: (b, m, n * block_n)),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel", "parallel")
        ),
    )(x, gate_kernel, up_kernel, down_kernel)
