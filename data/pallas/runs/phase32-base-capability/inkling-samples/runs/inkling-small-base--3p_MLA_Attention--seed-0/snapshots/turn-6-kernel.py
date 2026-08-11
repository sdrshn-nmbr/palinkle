import jax
import jax.numpy as jnp
import jax.experimental.pallas as pl

CONFIG = {
    "batch": 4,
    "emb_dim": 7168,
    "kv_lora_rank": 512,
    "model": "DeepSeek-V3-671B",
    "name": "deepseek_v3_mla",
    "num_heads": 128,
    "operator": "mla_attention",
    "q_lora_rank": 1536,
    "qk_nope_head_dim": 128,
    "qk_rope_head_dim": 64,
    "rope_theta": 10000,
    "seq_len": 2048,
    "v_head_dim": 128,
}

def _compute_rope(head_dim, seq_len, theta, dtype):
    freqs = 1.0 / (theta ** (jnp.arange(0, head_dim, 2, dtype=jnp.float32) / head_dim))
    pos = jnp.arange(seq_len, dtype=jnp.float32)
    angles = jnp.outer(pos, freqs)
    cos = jnp.cos(angles).astype(dtype)
    sin = jnp.sin(angles).astype(dtype)
    return cos, sin

def _apply_rope(x, cos, sin):
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    # Add newaxes to cos/sin for broadcasting: cos[None, :, None, :]
    cos = cos[None, :, None, :]
    sin = sin[None, :, None, :]
    rotated = jnp.stack([
        x1 * cos - x2 * sin,
        x1 * sin + x2 * cos,
    ], axis=-1)
    rotated = rotated.reshape(x.shape)
    return rotated

def workload(x, q_down_proj, q_up_proj, kv_down_proj, k_up_proj, v_up_proj, o_proj):
    C = CONFIG
    B, S, E = x.shape
    H = C["num_heads"]
    nope = C["qk_nope_head_dim"]
    rope = C["qk_rope_head_dim"]
    vd = C["v_head_dim"]
    kvl = C["kv_lora_rank"]
    
    def kernel(x_ref, q_down_ref, q_up_ref, kv_down_ref, k_up_ref, v_up_ref, o_proj_ref, out_ref):
        x_local = x_ref[...]
        q_down_local = q_down_ref[...]
        q_up_local = q_up_ref[...]
        kv_down_local = kv_down_ref[...]
        k_up_local = k_up_ref[...]
        v_up_local = v_up_ref[...]
        
        q = jnp.dot(x_local, q_down_local)
        q = jnp.dot(q, q_up_local)
        q = q.reshape(B, S, H, nope + rope)
        
        q_nope = q[..., :nope]
        q_rope = q[..., nope:]
        
        kv = jnp.dot(x_local, kv_down_local)
        k_latent = kv[..., :kvl]
        k_rope_raw = kv[..., kvl:]
        
        k_nope = jnp.dot(k_latent, k_up_local)
        k_nope = k_nope.reshape(B, S, H, nope)
        
        cos, sin = _compute_rope(rope, S, C["rope_theta"], x_local.dtype)
        
        # k_rope_raw -> (B, S, 1, rope) -> broadcast to (B, S, H, rope)
        k_rope = jnp.broadcast_to(k_rope_raw[..., None, :], (B, S, H, rope))
        k_rope = _apply_rope(k_rope, cos, sin)
        
        q_rope = _apply_rope(q_rope, cos, sin)
        
        v = jnp.dot(k_latent, v_up_local)
        v = v.reshape(B, S, H, vd)
        
        q_full = jnp.concatenate([q_nope, q_rope], axis=-1)
        q_full = jnp.transpose(q_full, (0, 2, 1, 3))
        
        k_full = jnp.concatenate([k_nope, k_rope], axis=-1)
        k_full = jnp.transpose(k_full, (0, 2, 1, 3))
        
        v = jnp.transpose(v, (0, 2, 1, 3))
        
        hd = nope + rope
        attn = jnp.einsum("bhqd,bhkd->bhqk", q_full, k_full)
        attn = attn * (hd ** -0.5)
        
        mask = jnp.tril(jnp.ones((S, S), dtype=jnp.float32))
        mask = jnp.broadcast_to(mask[None, None, :, :], attn.shape)
        attn = jnp.where(mask, attn, -1e9)
        
        attn = jax.nn.softmax(attn, axis=-1)
        
        out = jnp.einsum("bhqk,bhkd->bhqd", attn, v)
        out = jnp.transpose(out, (0, 2, 1, 3))
        out = out.reshape(B, S, H * vd)
        
        result = jnp.dot(out, o_proj_ref[...])
        out_ref[...] = result
    
    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        grid=(),
        in_specs=(
            pl.no_block_spec,
            pl.no_block_spec,
            pl.no_block_spec,
            pl.no_block_spec,
            pl.no_block_spec,
            pl.no_block_spec,
            pl.no_block_spec,
        ),
        out_specs=pl.no_block_spec,
    )(x, q_down_proj, q_up_proj, kv_down_proj, k_up_proj, v_up_proj, o_proj)
