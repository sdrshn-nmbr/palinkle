import jax
import jax.numpy as jnp
import jax.random as jrandom
import pallas as pl
import triton.pallas as pltpu

# Configuration from instruction.md
CONFIG = {
    "head_dim": 128,
    "max_seq_len": 4096,
    "model": "Llama-3.1-70B",
    "name": "llama3_70b_paged_attention",
    "num_kv_heads": 8,
    "num_query_heads": 64,
    "num_seqs": 64,
    "operator": "paged_attention",
    "page_size": 16,
    "pages_per_seq": 256
}


def workload(queries, k_pages, v_pages, kv_lens, page_indices, cu_q_lens):
    """Paged attention kernel for Llama-3.1-70B inference decode.
    
    Args:
        queries: [num_seqs, max_num_queries, head_dim] - query vectors
        k_pages: [num_pages, page_size, num_kv_heads, head_dim] - key pages
        v_pages: [num_pages, page_size, num_kv_heads, head_dim] - value pages
        kv_lens: [num_seqs] - actual KV lengths per sequence
        page_indices: [num_seqs, pages_per_seq] - page indices per sequence
        cu_q_lens: [num_seqs + 1] - cumulative query lengths
    
    Returns:
        [num_seqs, max_num_queries, head_dim] - attention outputs
    """
    num_seqs = CONFIG["num_seqs"]
    num_q_heads = CONFIG["num_query_heads"]
    num_kv_heads = CONFIG["num_kv_heads"]
    head_dim = CONFIG["head_dim"]
    page_size = CONFIG["page_size"]
    pages_per_seq = CONFIG["pages_per_seq"]
    
    num_q_per_kv = num_q_heads // num_kv_heads  # 64 // 8 = 8
    max_seq_len = pages_per_seq * page_size  # 256 * 16 = 4096
    sm_scale = head_dim ** -0.5  # 1/sqrt(head_dim)
    
    def paged_attention_kernel(ref_queues, ref_k_pages, ref_v_pages, 
                                ref_kv_lens, ref_page_indices, ref_cu_q_lens,
                                ref_outputs):
        """Pallas kernel for paged attention on one sequence."""
        seq_idx = pl.program_id(0)
        
        # Get query range for this sequence
        q_start = ref_cu_q_lens[seq_idx]
        q_end = ref_cu_q_lens[seq_idx + 1]
        num_queries = q_end - q_start
        
        # Get KV length for this sequence
        kv_len = ref_kv_lens[seq_idx]
        
        # Get page indices for this sequence
        seq_pages = ref_page_indices[seq_idx, :]
        
        # Gather K and V pages
        # k_pages: [num_pages, page_size, num_kv_heads, head_dim]
        # We need to gather pages and reshape to [max_seq_len, num_kv_heads, head_dim]
        k_gathered = ref_k_pages[seq_pages, :, :, :]  # [pages_per_seq, page_size, num_kv_heads, head_dim]
        v_gathered = ref_v_pages[seq_pages, :, :, :]  # [pages_per_seq, page_size, num_kv_heads, head_dim]
        
        # Reshape to [max_seq_len, num_kv_heads, head_dim]
        # [pages_per_seq, page_size, num_kv_heads, head_dim] -> [max_seq_len, num_kv_heads, head_dim]
        k_reshaped = k_gathered.reshape(max_seq_len, num_kv_heads, head_dim)
        v_reshaped = v_gathered.reshape(max_seq_len, num_kv_heads, head_dim)
        
        # Repeat for GQA: [max_seq_len, num_kv_heads, head_dim] -> [max_seq_len, num_q_heads, head_dim]
        k_repeat = jnp.repeat(k_reshaped, num_q_per_kv, axis=1)
        v_repeat = jnp.repeat(v_reshaped, num_q_per_kv, axis=1)
        
        # Get query for this sequence
        q = ref_queues[seq_idx, q_start:q_end, :]  # [num_queries, num_q_heads, head_dim]
        
        # Compute attention scores: q @ k.T * sm_scale
        # q: [num_queries, num_q_heads, head_dim]
        # k_repeat: [max_seq_len, num_q_heads, head_dim]
        # attn: [num_queries, num_q_heads, max_seq_len]
        attn = jnp.einsum('qhd,khd->hqk', q, k_repeat) * sm_scale
        
        # Create causal mask
        # mask: [max_seq_len] - True where position < kv_len
        mask = jnp.arange(max_seq_len) < kv_len
        
        # Apply mask: set positions beyond kv_len to -1e30
        attn = jnp.where(mask, attn, -1e30)
        
        # Softmax along last axis
        attn = jax.nn.softmax(attn, axis=-1)
        
        # Compute output: attn @ v
        # attn: [num_queries, num_q_heads, max_seq_len]
        # v_repeat: [max_seq_len, num_q_heads, head_dim]
        # out: [num_queries, num_q_heads, head_dim]
        out = jnp.einsum('hqk,khd->qhd', attn, v_repeat)
        
        # Write output
        ref_outputs[seq_idx, q_start:q_end, :] = out
    
    # Grid: one block per sequence
    grid = (num_seqs,)
    
    # Define block shapes
    # For queries: [num_seqs, max_num_queries, head_dim]
    # We'll use a block that covers all queries for a sequence
    # Actually, we need to handle variable-length queries per sequence
    
    # Let's use a simpler approach: process all sequences in parallel
    # with a grid that iterates over sequences
    
    def kernel(ref_queues, ref_k_pages, ref_v_pages, 
               ref_kv_lens, ref_page_indices, ref_cu_q_lens,
               ref_outputs):
        seq_idx = pl.program_id(0)
        
        # Get query range for this sequence
        q_start = ref_cu_q_lens[seq_idx]
        q_end = ref_cu_q_lens[seq_idx + 1]
        num_queries = q_end - q_start
        
        # Get KV length for this sequence
        kv_len = ref_kv_lens[seq_idx]
        
        # Get page indices for this sequence
        seq_pages = ref_page_indices[seq_idx, :]
        
        # Gather K and V pages
        k_gathered = ref_k_pages[seq_pages, :, :, :]
        v_gathered = ref_v_pages[seq_pages, :, :, :]
        
        # Reshape to [max_seq_len, num_kv_heads, head_dim]
        k_reshaped = k_gathered.reshape(max_seq_len, num_kv_heads, head_dim)
        v_reshaped = v_gathered.reshape(max_seq_len, num_kv_heads, head_dim)
        
        # Repeat for GQA
        k_repeat = jnp.repeat(k_reshaped, num_q_per_kv, axis=1)
        v_repeat = jnp.repeat(v_reshaped, num_q_per_kv, axis=1)
        
        # Get query for this sequence
        q = ref_queues[seq_idx, q_start:q_end, :]
        
        # Compute attention scores
        attn = jnp.einsum('qhd,khd->hqk', q, k_repeat) * sm_scale
        
        # Create causal mask
        mask = jnp.arange(max_seq_len) < kv_len
        
        # Apply mask
        attn = jnp.where(mask, attn, -1e30)
        
        # Softmax
        attn = jax.nn.softmax(attn, axis=-1)
        
        # Compute output
        out = jnp.einsum('hqk,khd->qhd', attn, v_repeat)
        
        # Write output
        ref_outputs[seq_idx, q_start:q_end, :] = out
    
    # Use pallas_call with appropriate specs
    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((num_seqs, 64, head_dim), queries.dtype),
        grid=grid,
        in_specs=(
            pl.BlockSpec((1, 64, head_dim), lambda i: (i, 0, 0)),  # queries
            pl.BlockSpec((pages_per_seq, page_size, num_kv_heads, head_dim), lambda i: (0, 0, 0, 0)),  # k_pages
            pl.BlockSpec((pages_per_seq, page_size, num_kv_heads, head_dim), lambda i: (0, 0, 0, 0)),  # v_pages
            pl.BlockSpec((num_seqs,), lambda i: (i,)),  # kv_lens
            pl.BlockSpec((num_seqs, pages_per_seq), lambda i: (i, 0)),  # page_indices
            pl.BlockSpec((num_seqs + 1,), lambda i: (i,)),  # cu_q_lens
        ),
        out_specs=pl.BlockSpec((1, 64, head_dim), lambda i: (i, 0, 0)),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel",)
        ),
    )(queries, k_pages, v_pages, kv_lens, page_indices, cu_q_lens)
