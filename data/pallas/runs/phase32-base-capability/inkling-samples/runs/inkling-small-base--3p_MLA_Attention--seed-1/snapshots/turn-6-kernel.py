import jax
import jax.numpy as jnp
import jax.experimental.pallas as pl

CONFIG = {
    "batch": 4,
    "emb_dim": 7168,
    "kv_lora_rank": 512,
    "num_heads": 128,
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
    return (jnp.cos(angles).astype(dtype), jnp.sin(angles).astype(dtype))

def _apply_rope(x, cos, sin):
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    # Subscript cos/sin as in AST: [None, :, None, :]
    cos = cos[None, :, None, :]
    sin = sin[None, :, None, :]
    rotated = jnp.stack([
        x1 * cos - x2 * sin,
        x1 * sin + x2 * cos
    ], axis=-1)
    return rotated.reshape(x.shape)

def mla_kernel(x_ref, q_down_proj_ref, q_up_proj_ref, kv_down_proj_ref, k_up_proj_ref, v_up_proj_ref, o_proj_ref, out_ref):
    x = x_ref[...]
    q_down_proj = q_down_proj_ref[...]
    q_up_proj = q_up_proj_ref[...]
    kv_down_proj = kv_down_proj_ref[...]
    k_up_proj = k_up_proj_ref[...]
    v_up_proj = v_up_proj_ref[...]
    o_proj = o_proj_ref[...]

    C = CONFIG
    B, S, E = x.shape
    H = C["num_heads"]
    nope, rope, vd = C["qk_nope_head_dim"], C["qk_rope_head_dim"], C["v_head_dim"]
    kvl = C["kv_lora_rank"]

    q = jnp.dot(x, q_down_proj)
    q = jnp.dot(q, q_up_proj)
    q = q.reshape(B, S, H, nope + rope)

    q_nope = q[..., :nope]
    q_rope = q[..., nope:]

    kv = jnp.dot(x, kv_down_proj)
    k_latent = kv[..., :kvl]
    k_rope_raw = kv[..., kvl:]

    k_nope = jnp.dot(k_latent, k_up_proj).reshape(B, S, H, nope)

    cos, sin = _compute_rope(rope, S, C["rope_theta"], x.dtype)
    # k_rope_raw is (B, S, rope). Broadcast to (B, S, H, rope)
    k_rope = jnp.broadcast_to(k_rope_raw[..., None, :], (B, S, H, rope))

    q_rope = _apply_rope(q_rope, cos, sin)
    k_rope = _apply_rope(k_rope, cos, sin)

    v = jnp.dot(k_latent, v_up_proj).reshape(B, S, H, vd)
    v = jnp.transpose(v, (0, 2, 1, 3))

    q_full = jnp.concatenate([q_nope, q_rope], axis=-1)
    q_full = jnp.transpose(q_full, (0, 2, 1, 3))
    k_full = jnp.concatenate([k_nope, k_rope], axis=-1)
    k_full = jnp.transpose(k_full, (0, 2, 1, 3))

    hd = nope + rope
    attn = jnp.einsum("bhqd,bhkd->bhqk", q_full, k_full) * (hd ** -0.5)

    mask = jnp.tril(jnp.ones((S, S), dtype=jnp.float32))
    mask = jnp.broadcast_to(mask[None, None, :, :], (B, H, S, S))
    attn = jnp.where(mask, attn, -1e9)
    attn = jax.nn.softmax(attn, axis=-1)

    out = jnp.einsum("bhqk,bhkd->bhqd", attn, v)
    out = jnp.transpose(out, (0, 2, 1, 3))
    out = out.reshape(B, S, H * vd)
    out = jnp.dot(out, o_proj)
    out_ref[...] = out

def workload(x, q_down_proj, q_up_proj, kv_down_proj, k_up_proj, v_up_proj, o_proj):
    return pl.pallas_call(
        mla_kernel,
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
