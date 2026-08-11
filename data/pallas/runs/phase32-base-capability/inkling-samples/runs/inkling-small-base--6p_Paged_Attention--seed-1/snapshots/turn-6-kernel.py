import jax
import jax.numpy as jnp
import jax.lax as lax
from jax import random
import jax.experimental.pallas as pl
import jax.experimental.pallas.tpu as pltpu

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

def workload(queries, k_pages, v_pages, kv_lens, page_indices, cu_q_lens):
    num_seqs = CONFIG["num_seqs"]
    num_q_heads = CONFIG["num_query_heads"]
    num_kv_heads = CONFIG["num_kv_heads"]
    head_dim = CONFIG["head_dim"]
    page_size = CONFIG["page_size"]
    max_seq_len = CONFIG["pages_per_seq"] * page_size
    pages_per_seq = CONFIG["pages_per_seq"]
    num_q_per_kv = num_q_heads // num_kv_heads
    sm_scale = head_dim ** -0.5

    def kernel(
        queries_ref,
        k_pages_ref,
        v_pages_ref,
        kv_lens_ref,
        page_indices_ref,
        cu_q_lens_ref,
        out_ref,
    ):
        s = pl.program_id(0)
        # Per-sequence query slice (decode: 1 token per seq)
        q_start = int(cu_q_lens_ref[s])
        q_end = int(cu_q_lens_ref[s + 1])
        # Dynamic slice of queries for this sequence
        q = lax.dynamic_slice(queries_ref[...], (q_start, 0, 0), (q_end - q_start, num_q_heads, head_dim))
        # Gather pages for this sequence
        seq_pages = page_indices_ref[s, :]
        # Index into page arrays; seq_pages is (pages_per_seq,)
        k = k_pages_ref[seq_pages, :, :, :]
        v = v_pages_ref[seq_pages, :, :, :]
        # Reshape to (max_seq_len, num_kv_heads, head_dim)
        k = jnp.reshape(k, (max_seq_len, num_kv_heads, head_dim))
        v = jnp.reshape(v, (max_seq_len, num_kv_heads, head_dim))
        # Repeat for GQA
        k = jnp.repeat(k, num_q_per_kv, axis=1)
        v = jnp.repeat(v, num_q_per_kv, axis=1)
        # Attention scores
        attn = jnp.einsum("qhd,khd->hqk", q, k) * sm_scale
        # Mask
        kv_len = int(kv_lens_ref[s])
        mask = jnp.arange(max_seq_len) < kv_len
        attn = jnp.where(mask[None, None, :], attn, -1e30)
        # Softmax over key length
        attn = jax.nn.softmax(attn, axis=-1)
        # Apply to values
        out = jnp.einsum("hqk,khd->qhd", attn, v)
        # Write back, squeezing the sequence token dimension
        out_ref[...] = jnp.squeeze(out, axis=0)

    out_shape = jax.ShapeDtypeStruct(queries.shape, queries.dtype)
    return pl.pallas_call(
        kernel,
        out_shape=out_shape,
        grid=(num_seqs,),
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
    )(queries, k_pages, v_pages, kv_lens, page_indices, cu_q_lens)
