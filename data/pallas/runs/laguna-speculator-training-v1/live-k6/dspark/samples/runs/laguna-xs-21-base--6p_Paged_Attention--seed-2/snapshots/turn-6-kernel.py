import jax
import jax.numpy as jnp
import jax.lax as lax
import jax.nn as nn
import pallas as pl
import pallas.triton as pltpu

# Configuration
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
        queries: [num_seqs, num_q_heads, head_dim] - query vectors
        k_pages: [total_pages, page_size, num_kv_heads, head_dim] - key pages
        v_pages: [total_pages, page_size, num_kv_heads, head_dim] - value pages
        kv_lens: [num_seqs] - actual KV lengths per sequence
        page_indices: [num_seqs, pages_per_seq] - page indices per sequence
        cu_q_lens: [num_seqs + 1] - cumulative query lengths
    
    Returns:
        [num_seqs, num_q_heads, head_dim] - attention outputs
    """
    num_seqs = CONFIG["num_seqs"]
    num_q_heads = CONFIG["num_query_heads"]
    num_kv_heads = CONFIG["num_kv_heads"]
    head_dim = CONFIG["head_dim"]
    page_size = CONFIG["page_size"]
    pages_per_seq = CONFIG["pages_per_seq"]
    max_seq_len = pages_per_seq * page_size  # 4096
    num_q_per_kv = num_q_heads // num_kv_heads  # 8
    sm_scale = head_dim ** (-0.5)
    
    def attend_one_seq(seq_idx):
        """Compute attention for a single sequence."""
        # Get query range for this sequence
        q_start = cu_q_lens[seq_idx]
        q_end = cu_q_lens[seq_idx + 1]
        
        # Extract query slice [num_queries, num_q_heads, head_dim]
        q = lax.dynamic_slice(
            queries,
            (q_start, 0, 0),
            (q_end - q_start, num_q_heads, head_dim)
        )
        
        # Get page indices for this sequence
        seq_pages = page_indices[seq_idx]  # [pages_per_seq]
        
        # Gather K and V pages and reshape to [max_seq_len, num_kv_heads, head_dim]
        k = k_pages[seq_pages].reshape(max_seq_len, num_kv_heads, head_dim)
        v = v_pages[seq_pages].reshape(max_seq_len, num_kv_heads, head_dim)
        
        # Repeat for GQA: [max_seq_len, num_q_heads, head_dim]
        k = jnp.repeat(k, num_q_per_kv, axis=1)
        v = jnp.repeat(v, num_q_per_kv, axis=1)
        
        # Compute attention scores: [num_queries, num_q_heads, num_q_heads]
        attn = jnp.einsum('qhd,khd->hqk', q, k) * sm_scale
        
        # Get actual KV length for this sequence
        kv_len = kv_lens[seq_idx]
        
        # Create causal mask: [max_seq_len]
        mask = jnp.arange(max_seq_len) < kv_len
        
        # Apply mask with large negative value
        attn = jnp.where(mask, attn, -1e30)
        
        # Softmax along last axis
        attn = nn.softmax(attn, axis=-1)
        
        # Compute output: [num_queries, num_q_heads, head_dim]
        out = jnp.einsum('hqk,khd->qhd', attn, v)
        
        return out
    
    # Vectorize over sequences
    outputs = jax.vmap(attend_one_seq)(jnp.arange(num_seqs))
    
    return outputs


def _attend_one_seq_kernel(seq_idx_ref, queries_ref, k_pages_ref, v_pages_ref, 
                           kv_lens_ref, page_indices_ref, cu_q_lens_ref, out_ref):
    """Pallas kernel for attention on one sequence."""
    # Configuration
    head_dim = 128
    num_q_heads = 64
    num_kv_heads = 8
    page_size = 16
    pages_per_seq = 256
    max_seq_len = pages_per_seq * page_size
    num_q_per_kv = 8
    sm_scale = head_dim ** (-0.5)
    
    seq_idx = seq_idx_ref[()]
    
    # Get query range
    q_start = cu_q_lens_ref[seq_idx]
    q_end = cu_q_lens_ref[seq_idx + 1]
    num_queries = q_end - q_start
    
    # Get page indices
    seq_pages = page_indices_ref[seq_idx, :]  # [pages_per_seq]
    
    # Gather K and V pages
    # k_pages: [total_pages, page_size, num_kv_heads, head_dim]
    # seq_pages: [pages_per_seq]
    # Result should be [max_seq_len, num_kv_heads, head_dim]
    k_gathered = k_pages_ref[seq_pages, :, :, :]  # [pages_per_seq, page_size, num_kv_heads, head_dim]
    v_gathered = v_pages_ref[seq_pages, :, :, :]
    
    # Reshape to [max_seq_len, num_kv_heads, head_dim]
    k = k_gathered.reshape(max_seq_len, num_kv_heads, head_dim)
    v = v_gathered.reshape(max_seq_len, num_kv_heads, head_dim)
    
    # Repeat for GQA
    k = jnp.repeat(k, num_q_per_kv, axis=1)  # [max_seq_len, num_q_heads, head_dim]
    v = jnp.repeat(v, num_q_per_kv, axis=1)
    
    # Get KV length
    kv_len = kv_lens_ref[seq_idx]
    
    # Extract query for this sequence
    q = queries_ref[q_start:q_end, :, :]  # [num_queries, num_q_heads, head_dim]
    
    # Compute attention scores
    attn = jnp.einsum('qhd,khd->hqk', q, k) * sm_scale  # [num_queries, num_q_heads, num_q_heads]
    
    # Create causal mask
    mask = jnp.arange(max_seq_len) < kv_len  # [max_seq_len]
    
    # Apply mask
    attn = jnp.where(mask, attn, -1e30)
    
    # Softmax
    attn = nn.softmax(attn, axis=-1)
    
    # Compute output
    out = jnp.einsum('hqk,khd->qhd', attn, v)  # [num_queries, num_q_heads, head_dim]
    
    # Write output
    out_ref[q_start:q_end, :, :] = out


def workload_pallas(queries, k_pages, v_pages, kv_lens, page_indices, cu_q_lens):
    """Paged attention using Pallas kernel."""
    num_seqs = CONFIG["num_seqs"]
    num_q_heads = CONFIG["num_query_heads"]
    head_dim = CONFIG["head_dim"]
    
    def kernel(seq_idx_ref, queries_ref, k_pages_ref, v_pages_ref,
               kv_lens_ref, page_indices_ref, cu_q_lens_ref, out_ref):
        _attend_one_seq_kernel(seq_idx_ref, queries_ref, k_pages_ref, v_pages_ref,
                               kv_lens_ref, page_indices_ref, cu_q_lens_ref, out_ref)
    
    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((num_seqs, num_q_heads, head_dim), queries.dtype),
        grid=(num_seqs,),
        in_specs=(
            pl.BlockSpec((1,), lambda i: (i,)),  # seq_idx
            pl.BlockSpec((None, num_q_heads, head_dim), lambda i: (i, 0, 0)),  # queries
            pl.BlockSpec((None, 16, 8, head_dim), lambda i: (i, 0, 0, 0)),  # k_pages
            pl.BlockSpec((None, 16, 8, head_dim), lambda i: (i, 0, 0, 0)),  # v_pages
            pl.BlockSpec((None,), lambda i: (i,)),  # kv_lens
            pl.BlockSpec((None, 256), lambda i: (i, 0)),  # page_indices
            pl.BlockSpec((None,), lambda i: (i,)),  # cu_q_lens
        ),
        out_specs=pl.BlockSpec((None, num_q_heads, head_dim), lambda i: (i, 0, 0)),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel",)
        ),
    )(jnp.arange(num_seqs), queries, k_pages, v_pages, kv_lens, page_indices, cu_q_lens)


# Use the vmap-based implementation as the main workload
# This is the standard JAX approach that Pallas will lower
workload = workload
