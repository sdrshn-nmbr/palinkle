import jax
import jax.numpy as jnp
import jax.lax as lax
import jax.experimental.pallas as pl
import jax.experimental.pallas.tpu as pltpu

def workload(x, q_down_proj, q_up_proj, kv_down_proj, k_up_proj, v_up_proj, o_proj):
    # Configuration
    B, S, emb_dim = x.shape
    H = 128
    nope = 128
    rope = 64
    vd = 128
    kvl = 512
    hd = nope + rope
    
    def kernel(x_ref, q_down_proj_ref, q_up_proj_ref, kv_down_proj_ref, k_up_proj_ref, v_up_proj_ref, o_proj_ref, out_ref):
        x = x_ref[...]
        q_down_proj = q_down_proj_ref[...]
        q_up_proj = q_up_proj_ref[...]
        kv_down_proj = kv_down_proj_ref[...]
        k_up_proj = k_up_proj_ref[...]
        v_up_proj = v_up_proj_ref[...]
        o_proj = o_proj_ref[...]
        
        # q projection
        q = jnp.dot(x, q_down_proj)
        q = jnp.dot(q, q_up_proj)
        q = q.reshape(B, S, H, hd)
        
        q_nope = q[..., :nope]
        q_rope = q[..., nope:]
        
        # kv projection
        kv = jnp.dot(x, kv_down_proj)
        k_latent = kv[..., :kvl]
        k_rope_raw = kv[..., kvl:]
        
        k_nope = jnp.dot(k_latent, k_up_proj).reshape(B, S, H, nope)
        
        # k_rope broadcast
        k_rope_raw_reshaped = k_rope_raw.reshape(B, S, 1, rope)
        k_rope = jnp.broadcast_to(k_rope_raw_reshaped, (B, S, H, rope))
        
        # rope computation
        def _compute_rope(head_dim, seq_len, theta, dtype):
            freqs = 1.0 / (theta ** (jnp.arange(0, head_dim, 2, dtype=jnp.float32) / head_dim))
            pos = jnp.arange(seq_len, dtype=jnp.float32)
            angles = jnp.outer(pos, freqs)
            cos = jnp.cos(angles).astype(dtype)
            sin = jnp.sin(angles).astype(dtype)
            return cos, sin
        
        cos, sin = _compute_rope(rope, S, 10000.0, x.dtype)
        
        def _apply_rope(x, cos, sin):
            x1 = x[..., ::2]
            x2 = x[..., 1::2]
            cos = cos[..., :, :]
            sin = sin[..., :, :]
            # Need to match shapes: cos/sin are (S, rope/2) or similar
            # Actually cos/sin from _compute_rope are (seq_len, head_dim/2)
            # We need to broadcast to (B, S, H, rope/2)
            cos = jnp.broadcast_to(cos[None, :, None, :], (B, S, H, rope//2))
            sin = jnp.broadcast_to(sin[None, :, None, :], (B, S, H, rope//2))
            rotated1 = x1 * cos - x2 * sin
            rotated2 = x1 * sin + x2 * cos
            rotated = jnp.stack([rotated1, rotated2], axis=-1)
            rotated = rotated.reshape(x.shape)
            return rotated
        
        q_rope = _apply_rope(q_rope, cos, sin)
        k_rope = _apply_rope(k_rope, cos, sin)
        
        # v projection
        v = jnp.dot(k_latent, v_up_proj).reshape(B, S, H, vd)
        
        # concatenate q and k
        q_full = jnp.concatenate([q_nope, q_rope], axis=-1)
        q_full = jnp.transpose(q_full, (0, 2, 1, 3))  # (B, H, S, hd)
        
        k_full = jnp.concatenate([k_nope, k_rope], axis=-1)
        k_full = jnp.transpose(k_full, (0, 2, 1, 3))  # (B, H, S, hd)
        
        v = jnp.transpose(v, (0, 2, 1, 3))  # (B, H, S, vd)
        
        # attention scores
        attn = jnp.einsum("bhqd,bhkd->bhqk", q_full, k_full)
        attn = attn * (hd ** -0.5)
        
        # causal mask
        mask = jnp.tril(jnp.ones((S, S), dtype=jnp.float32))
        attn = jnp.where(mask, attn, -1e9)
        
        # softmax
        attn = jax.nn.softmax(attn, axis=-1)
        
        # apply attention to v
        out = jnp.einsum("bhqk,bhkd->bhqd", attn, v)
        
        # reshape and transpose back
        out = jnp.transpose(out, (0, 2, 1, 3))  # (B, S, H, vd)
        out = out.reshape(B, S, H * vd)
        
        # final projection
        out = jnp.dot(out, o_proj)
        
        out_ref[...] = out
    
    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        grid=(),
        in_specs=[pl.no_block_spec] * 7,
        out_specs=pl.no_block_spec,
    )(x, q_down_proj, q_up_proj, kv_down_proj, k_up_proj, v_up_proj, o_proj)
