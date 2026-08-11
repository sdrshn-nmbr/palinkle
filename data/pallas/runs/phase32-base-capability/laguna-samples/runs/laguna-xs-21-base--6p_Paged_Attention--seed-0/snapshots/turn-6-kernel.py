import jax
import jax.numpy as jnp
import jax.random as jrandom
import pallas as pl
import triton.pallas as pltpu

# Configuration from the spec
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
    "pages_per_seq": 256,
}

# Constants
NUM_SEQS = CONFIG["num_seqs"]
NUM_Q_HEADS = CONFIG["num_query_heads"]
NUM_KV_HEADS = CONFIG["num_kv_heads"]
HEAD_DIM = CONFIG["head_dim"]
PAGE_SIZE = CONFIG["page_size"]
PAGES_PER_SEQ = CONFIG["pages_per_seq"]
MAX_SEQ_LEN = CONFIG["max_seq_len"]
SM_SCALE = (HEAD_DIM ** -0.5)  # 1/sqrt(head_dim)


def paged_attention_kernel(
    queries_ref,
    k_pages_ref,
    v_pages_ref,
    kv_lens_ref,
    page_indices_ref,
    cu_q_lens_ref,
    outputs_ref,
):
    """Pallas kernel for paged attention."""
    seq_idx = pl.program_id(0)
    
    # Get q_start and q_end from cu_q_lens
    q_start = cu_q_lens_ref[seq_idx]
    q_end = cu_q_lens_ref[seq_idx + 1]
    
    # Get kv_len for this sequence
    kv_len = kv_lens_ref[seq_idx]
    
    # Get seq_pages from page_indices
    seq_pages = page_indices_ref[seq_idx]  # Shape: [pages_per_seq]
    
    # Gather k and v pages
    # k_pages and v_pages have shape [num_seqs * pages_per_seq, page_size, num_kv_heads, head_dim]
    # We need to gather the pages for this sequence
    k_gathered = k_pages_ref[seq_pages]  # Shape: [pages_per_seq, page_size, num_kv_heads, head_dim]
    v_gathered = v_pages_ref[seq_pages]  # Shape: [pages_per_seq, page_size, num_kv_heads, head_dim]
    
    # Reshape to [max_seq_len, num_kv_heads, head_dim]
    # Flatten the pages and page_size dimensions
    k_flat = jnp.reshape(k_gathered, (MAX_SEQ_LEN, NUM_KV_HEADS, HEAD_DIM))
    v_flat = jnp.reshape(v_gathered, (MAX_SEQ_LEN, NUM_KV_HEADS, HEAD_DIM))
    
    # Get query slice for this sequence
    # queries have shape [num_seqs, num_query_heads, head_dim]
    q = queries_ref[seq_idx]  # Shape: [num_query_heads, head_dim]
    
    # Repeat k and v for GQA (grouped query attention)
    # num_q_per_kv = num_query_heads // num_kv_heads
    num_q_per_kv = NUM_Q_HEADS // NUM_KV_HEADS
    
    # k and v need to be repeated along axis 1
    # Original shape: [max_seq_len, num_kv_heads, head_dim]
    # After repeat: [max_seq_len, num_kv_heads * num_q_per_kv, head_dim]
    k_repeat = jnp.repeat(k_flat, num_q_per_kv, axis=1)
    v_repeat = jnp.repeat(v_flat, num_q_per_kv, axis=1)
    
    # Compute attention scores
    # q: [num_query_heads, head_dim]
    # k_repeat: [max_seq_len, num_query_heads, head_dim]
    # attn: [max_seq_len, num_query_heads]
    
    # We need to compute q @ k.T for each position
    # q shape: [num_query_heads, head_dim]
    # k_repeat shape: [max_seq_len, num_query_heads, head_dim]
    
    # Transpose k for matmul
    k_T = jnp.transpose(k_repeat, (1, 2, 0))  # [num_query_heads, head_dim, max_seq_len]
    
    # Compute attention scores: q @ k.T
    # q: [num_query_heads, head_dim]
    # k_T: [num_query_heads, head_dim, max_seq_len]
    # result: [num_query_heads, max_seq_len]
    attn_scores = jnp.dot(q, k_T) * SM_SCALE  # [num_query_heads, max_seq_len]
    
    # Create mask for positions beyond kv_len
    # mask: [max_seq_len]
    positions = jnp.arange(MAX_SEQ_LEN)
    mask = positions < kv_len
    
    # Apply mask (set masked positions to -inf)
    attn_scores = jnp.where(mask, attn_scores, -1e30)
    
    # Softmax along the sequence dimension
    attn_probs = jax.nn.softmax(attn_scores, axis=1)  # [num_query_heads, max_seq_len]
    
    # Compute output: attn @ v
    # attn_probs: [num_query_heads, max_seq_len]
    # v_repeat: [max_seq_len, num_query_heads, head_dim]
    # result: [num_query_heads, head_dim]
    
    # Need to transpose v for correct matmul
    v_T = jnp.transpose(v_repeat, (1, 2, 0))  # [num_query_heads, head_dim, max_seq_len]
    
    # Output: attn_probs @ v_T
    # [num_query_heads, max_seq_len] @ [num_query_heads, head_dim, max_seq_len]
    # This needs careful handling
    
    # Actually, let's do it differently
    # attn_probs: [num_query_heads, max_seq_len]
    # v_repeat: [max_seq_len, num_query_heads, head_dim]
    
    # For each query head, we compute weighted sum over sequence positions
    # output[i, j] = sum_k attn_probs[i, k] * v_repeat[k, j, :]
    
    # Let's use einsum
    output = jnp.einsum('qk,qhd->hd', attn_probs, v_repeat)
    
    # Store output
    outputs_ref[seq_idx] = output


def workload(queries, k_pages, v_pages, kv_lens, page_indices, cu_q_lens):
    """Paged attention workload."""
    # Grid over sequences
    grid = (NUM_SEQS,)
    
    # Define block spec for each input
    # queries: [num_seqs, num_query_heads, head_dim]
    # We process one sequence at a time
    
    def index_map(seq_idx):
        return (seq_idx,)
    
    # For simplicity, let's use a more direct approach
    # Each sequence is processed independently
    
    # Create the kernel
    def kernel_impl(
        queries_ref,
        k_pages_ref,
        v_pages_ref,
        kv_lens_ref,
        page_indices_ref,
        cu_q_lens_ref,
        outputs_ref,
    ):
        seq_idx = pl.program_id(0)
        
        # Get q_start and q_end from cu_q_lens
        q_start = cu_q_lens_ref[seq_idx]
        q_end = cu_q_lens_ref[seq_idx + 1]
        
        # Get kv_len for this sequence
        kv_len = kv_lens_ref[seq_idx]
        
        # Get seq_pages from page_indices
        seq_pages = page_indices_ref[seq_idx]  # Shape: [pages_per_seq]
        
        # Gather k and v pages
        k_gathered = k_pages_ref[seq_pages]  # Shape: [pages_per_seq, page_size, num_kv_heads, head_dim]
        v_gathered = v_pages_ref[seq_pages]  # Shape: [pages_per_seq, page_size, num_kv_heads, head_dim]
        
        # Reshape to [max_seq_len, num_kv_heads, head_dim]
        k_flat = jnp.reshape(k_gathered, (MAX_SEQ_LEN, NUM_KV_HEADS, HEAD_DIM))
        v_flat = jnp.reshape(v_gathered, (MAX_SEQ_LEN, NUM_KV_HEADS, HEAD_DIM))
        
        # Get query slice for this sequence
        q = queries_ref[seq_idx]  # Shape: [num_query_heads, head_dim]
        
        # Repeat k and v for GQA
        num_q_per_kv = NUM_Q_HEADS // NUM_KV_HEADS
        k_repeat = jnp.repeat(k_flat, num_q_per_kv, axis=1)
        v_repeat = jnp.repeat(v_flat, num_q_per_kv, axis=1)
        
        # Compute attention scores
        # q: [num_query_heads, head_dim]
        # k_repeat: [max_seq_len, num_query_heads, head_dim]
        k_T = jnp.transpose(k_repeat, (1, 2, 0))  # [num_query_heads, head_dim, max_seq_len]
        attn_scores = jnp.dot(q, k_T) * SM_SCALE  # [num_query_heads, max_seq_len]
        
        # Create mask
        positions = jnp.arange(MAX_SEQ_LEN)
        mask = positions < kv_len
        
        # Apply mask
        attn_scores = jnp.where(mask, attn_scores, -1e30)
        
        # Softmax
        attn_probs = jax.nn.softmax(attn_scores, axis=1)  # [num_query_heads, max_seq_len]
        
        # Compute output
        output = jnp.einsum('qk,khd->hd', attn_probs, v_repeat)
        
        outputs_ref[seq_idx] = output
    
    # Define block specs
    block_size = 1  # Process one sequence at a time
    
    return pl.pallas_call(
        kernel_impl,
        out_shape=jax.ShapeDtypeStruct((NUM_SEQS, NUM_Q_HEADS, HEAD_DIM), queries.dtype),
        grid=grid,
        in_specs=(
            pl.BlockSpec((1, NUM_Q_HEADS, HEAD_DIM), lambda i: (i, 0, 0)),  # queries
            pl.BlockSpec((PAGES_PER_SEQ, PAGE_SIZE, NUM_KV_HEADS, HEAD_DIM), lambda i: (0, 0, 0, 0)),  # k_pages
            pl.BlockSpec((PAGES_PER_SEQ, PAGE_SIZE, NUM_KV_HEADS, HEAD_DIM), lambda i: (0, 0, 0, 0)),  # v_pages
            pl.BlockSpec((NUM_SEQS,), lambda i: (i,)),  # kv_lens
            pl.BlockSpec((NUM_SEQS, PAGES_PER_SEQ), lambda i: (i, 0)),  # page_indices
            pl.BlockSpec((NUM_SEQS + 1,), lambda i: (0,)),  # cu_q_lens
        ),
        out_specs=pl.BlockSpec((NUM_SEQS, NUM_Q_HEADS, HEAD_DIM), lambda i: (i, 0, 0)),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("sequential",)
        ),
    )(queries, k_pages, v_pages, kv_lens, page_indices, cu_q_lens)
