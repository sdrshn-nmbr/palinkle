import jax
import jax.numpy as jnp
import jax.nn as nn
import pallas as pl
import triton.pallas as pltpu

# Configuration
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
    "v_head_dim": 128
}


def _compute_rope(head_dim, seq_len, theta, dtype):
    """Compute cosine and sine for rotary position embeddings."""
    freqs = 1.0 / (theta ** (jnp.arange(0, head_dim, 2) / head_dim))
    pos = jnp.arange(seq_len)
    angles = jnp.outer(pos, freqs)
    return jnp.cos(angles).astype(dtype), jnp.sin(angles).astype(dtype)


def _apply_rope(x, cos, sin):
    """Apply rotary position embeddings to input."""
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    rotated = jnp.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], axis=-1)
    return rotated.reshape(x.shape)


def workload(x, q_down_proj, q_up_proj, kv_down_proj, k_up_proj, v_up_proj, o_proj):
    """
    Multi-head Latent Attention (MLA) kernel for DeepSeek V3 671B.
    
    Args:
        x: Input tensor [B, S, E]
        q_down_proj: Q projection down [E, q_lora_rank]
        q_up_proj: Q projection up [q_lora_rank, H * (nope + rope)]
        kv_down_proj: KV projection down [E, 2 * kv_lora_rank]
        k_up_proj: K projection up [kv_lora_rank, H * nope]
        v_up_proj: V projection up [kv_lora_rank, H * v_head_dim]
        o_proj: Output projection [H * (nope + rope), E]
    
    Returns:
        Output tensor [B, S, E]
    """
    C = CONFIG
    B, S, E = x.shape
    H = C["num_heads"]
    nope = C["qk_nope_head_dim"]
    rope = C["qk_rope_head_dim"]
    vd = C["v_head_dim"]
    kvl = C["kv_lora_rank"]
    
    # Compute Q: (B, S, E) -> (B, S, q_lora_rank) -> (B, S, H * (nope + rope))
    q = jnp.dot(x, q_down_proj)
    q = jnp.dot(q, q_up_proj)
    
    # Reshape Q to [B, S, H, nope + rope]
    q = q.reshape(B, S, H, nope + rope)
    
    # Split Q into nope and rope parts
    q_nope = q[..., :nope]
    q_rope = q[..., nope:]
    
    # Compute KV: (B, S, E) -> (B, S, 2 * kv_lora_rank)
    kv = jnp.dot(x, kv_down_proj)
    
    # Split KV into k_latent and k_rope_raw
    k_latent = kv[..., :kvl]
    k_rope_raw = kv[..., kvl:]
    
    # Compute K_nope: (B, S, kv_lora_rank) -> (B, S, H * nope) -> reshape to [B, S, H, nope]
    k_nope = jnp.dot(k_latent, k_up_proj)
    k_nope = k_nope.reshape(B, S, H, nope)
    
    # Compute rope cos/sin
    cos, sin = _compute_rope(rope, S, C["rope_theta"], x.dtype)
    
    # Broadcast k_rope_raw to [B, S, H, rope]
    k_rope = jnp.broadcast_to(k_rope_raw, (B, S, H, rope))
    
    # Apply rope to q_rope and k_rope
    q_rope = _apply_rope(q_rope, cos, sin)
    k_rope = _apply_rope(k_rope, cos, sin)
    
    # Compute V: (B, S, kv_lora_rank) -> (B, S, H * v_head_dim) -> reshape to [B, S, H, vd]
    v = jnp.dot(k_latent, v_up_proj)
    v = v.reshape(B, S, H, vd)
    
    # Concatenate Q and K along last dimension, then transpose to [B, H, S, hd]
    q_full = jnp.transpose(jnp.concatenate([q_nope, q_rope], axis=-1), (0, 2, 1, 3))
    k_full = jnp.transpose(jnp.concatenate([k_nope, k_rope], axis=-1), (0, 2, 1, 3))
    
    # Transpose V to [B, H, S, vd]
    v = jnp.transpose(v, (0, 2, 1, 3))
    
    hd = nope + rope
    
    # Compute attention scores: (B, H, S, hd) @ (B, H, S, hd) -> (B, H, S, S)
    attn = jnp.einsum("bhqd,bhkd->bhqk", q_full, k_full) * (hd ** -0.5)
    
    # Create causal mask
    mask = jnp.tril(jnp.ones((S, S)))
    
    # Apply mask (masked positions get large negative value)
    attn = jnp.where(mask, attn, -1e9)
    
    # Softmax along last axis
    attn = nn.softmax(attn, axis=-1)
    
    # Compute output: (B, H, S, S) @ (B, H, S, vd) -> (B, H, S, vd)
    out = jnp.einsum("bhqk,bhkd->bhqd", attn, v)
    
    # Transpose back to [B, S, H, vd]
    out = jnp.transpose(out, (0, 2, 1, 3))
    
    # Reshape to [B, S, H * vd]
    out = out.reshape(B, S, H * vd)
    
    # Final projection
    return jnp.dot(out, o_proj)
