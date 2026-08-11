import jax
import jax.numpy as jnp
import jax.lax as lax
import math
import jax.experimental.pallas as pl
import jax.experimental.pallas.tpu as pltpu

CONFIG = {
    "head_dim": 128,
    "max_num_batched_tokens": 4096,
    "max_num_seqs": 64,
    "page_size": 16,
    "pages_per_seq": 256,
}

DEFAULT_MASK_VALUE = -0.7 * float(jnp.finfo(jnp.float32).max)

def workload(queries, kv_pages, kv_lens, page_indices, cu_q_lens, num_seqs):
    head_dim = CONFIG["head_dim"]
    max_seqs = CONFIG["max_num_seqs"]
    tokens_per_seq = CONFIG["max_num_batched_tokens"] // max_seqs
    num_q_heads = queries.shape[1]
    _, _, num_combined_kv_heads, _ = kv_pages.shape
    num_kv_heads = num_combined_kv_heads // 2
    num_query_per_kv = num_q_heads // num_kv_heads
    sm_scale = 1.0 / math.sqrt(head_dim)
    
    def kernel(queries_ref, kv_pages_ref, kv_lens_ref, page_indices_ref, cu_q_lens_ref, num_seqs_ref, out_ref):
        i = pl.program_id(0)
        # Read scalar values
        q_start = cu_q_lens_ref[i]
        kv_len = kv_lens_ref[i]
        indices = page_indices_ref[i, :]
        
        # Load query slice
        q = queries_ref[q_start:q_start + tokens_per_seq, :, :]
        
        # Load KV pages
        # indices is [256], we index first dim
        k_pages = kv_pages_ref[indices, 0::2, :, :]
        v_pages = kv_pages_ref[indices, 1::2, :, :]
        
        # Reshape to (-1, num_kv_heads, head_dim)
        k = jnp.reshape(k_pages, (-1, num_kv_heads, head_dim))
        v = jnp.reshape(v_pages, (-1, num_kv_heads, head_dim))
        
        # Repeat for query heads
        k = jnp.repeat(k, num_query_per_kv, axis=1)
        v = jnp.repeat(v, num_query_per_kv, axis=1)
        
        # Attention scores
        attn = jnp.einsum("qhd,khd->hqk", q, k, preferred_element_type=jnp.float32)
        attn = attn * sm_scale
        
        # Mask
        q_span = (kv_len - tokens_per_seq) + lax.broadcasted_iota(jnp.int32, jnp.shape(attn), 1)
        kv_span = lax.broadcasted_iota(jnp.int32, jnp.shape(attn), 2)
        mask = (q_span < kv_span) | (kv_span >= kv_len)
        attn = jnp.where(mask, DEFAULT_MASK_VALUE, attn)
        
        # Softmax
        attn = jax.nn.softmax(attn.astype(jnp.float32), axis=-1).astype(v.dtype)
        
        # Output
        out = jnp.einsum("hqk,khd->qhd", attn, v, preferred_element_type=jnp.float32)
        out = out.astype(queries_ref.dtype)
        
        # Valid check
        is_valid = i < num_seqs_ref[0]
        out = jnp.where(is_valid, out, 0.0)
        
        # Write to output slice
        out_ref[q_start:q_start + tokens_per_seq, :, :] = out
    
    out_shape = jax.ShapeDtypeStruct(queries.shape, queries.dtype)
    
    return pl.pallas_call(
        kernel,
        out_shape=out_shape,
        grid=(max_seqs,),
        in_specs=(
            pl.no_block_spec,
            pl.no_block_spec,
            pl.no_block_spec,
            pl.no_block_spec,
            pl.no_block_spec,
            pl.no_block_spec,
        ),
        out_specs=pl.no_block_spec,
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel",)),
    )(queries, kv_pages, kv_lens, page_indices, cu_q_lens, num_seqs)
