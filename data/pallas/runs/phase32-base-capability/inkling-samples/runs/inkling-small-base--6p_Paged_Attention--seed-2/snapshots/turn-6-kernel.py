import jax
import jax.numpy as jnp
import jax.experimental.pallas as pl
import jax.experimental.pallas.tpu as pltpu

def workload(queries, k_pages, v_pages, kv_lens, page_indices, cu_q_lens):
    # Gather paged KV for all sequences: page_indices [64, 256] -> gather pages
    # k_pages [16384, 16, 8, 128] -> [64, 256, 16, 8, 128] -> reshape [64, 4096, 8, 128]
    k_gathered = jnp.reshape(k_pages[page_indices], (64, 4096, 8, 128))
    v_gathered = jnp.reshape(v_pages[page_indices], (64, 4096, 8, 128))
    # GQA repeat: 64 query heads / 8 kv heads = 8
    k_gathered = jnp.repeat(k_gathered, 8, axis=2)
    v_gathered = jnp.repeat(v_gathered, 8, axis=2)
    
    # Mask: [64, 4096]
    max_seq_len = 4096
    mask = jnp.arange(max_seq_len)[None, :] < kv_lens[:, None]
    
    def kernel(q_ref, k_ref, v_ref, mask_ref, out_ref):
        b = pl.program_id(0)
        # q_ref block: [1, 64, 128] at (b, 0, 0) -> queries[b:b+1, :, :]
        q = q_ref[b, :, :]  # [64, 128]
        k = k_ref[b, :, :, :]  # [4096, 64, 128]
        v = v_ref[b, :, :, :]  # [4096, 64, 128]
        m = mask_ref[b, :]  # [4096]
        
        # Attention scores: q [64, 128], k [4096, 64, 128]
        scores = jnp.einsum("hd,shd->hs", q, k) * (128.0 ** -0.5)
        scores = jnp.where(m[None, :], scores, -1e30)
        attn = jax.nn.softmax(scores, axis=-1)
        out = jnp.einsum("hs,shd->hd", attn, v)
        out_ref[b, :, :] = out
    
    out_shape = jax.ShapeDtypeStruct(queries.shape, queries.dtype)
    return pl.pallas_call(
        kernel,
        out_shape=out_shape,
        grid=(64,),
        in_specs=(
            pl.BlockSpec((1, 64, 128), lambda b: (b, 0, 0)),
            pl.BlockSpec((1, 4096, 64, 128), lambda b: (b, 0, 0, 0)),
            pl.BlockSpec((1, 4096, 64, 128), lambda b: (b, 0, 0, 0)),
            pl.BlockSpec((1, 4096), lambda b: (b, 0)),
        ),
        out_specs=pl.BlockSpec((1, 64, 128), lambda b: (b, 0, 0)),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel",)),
    )(queries, k_gathered, v_gathered, mask)
