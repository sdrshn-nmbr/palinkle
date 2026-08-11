import jax
import jax.numpy as jnp
import jax.experimental.pallas as pl
import jax.experimental.pallas.tpu as pltpu

CONFIG = {"batch": 4, "head_dim": 128, "seq_len": 4096, "num_heads": 64}

D = CONFIG["head_dim"]
S = CONFIG["seq_len"]
B = CONFIG["batch"]
H = CONFIG["num_heads"]
sm_scale = D ** -0.5

block_q = 128
block_kv = 128
num_q_blocks = S // block_q
num_kv_blocks = S // block_kv

def flex_kernel(q_ref, k_ref, v_ref, rel_ref, score_ref, out_ref):
    bh = pl.program_id(0)
    q_idx = pl.program_id(1)
    b = bh // H
    h = bh % H
    q_start = q_idx * block_q
    q_end = q_start + block_q

    # Load query block
    q = q_ref[b, h, q_start:q_end, :]

    # Initialize score scratch to very negative
    score_ref[...] = jnp.full(score_ref.shape, -1e30, dtype=jnp.float32)

    # First pass: compute attention scores
    for kv_idx in range(num_kv_blocks):
        kv_start = kv_idx * block_kv
        kv_end = kv_start + block_kv
        k = k_ref[b, h, kv_start:kv_end, :]
        # Compute scores
        scores = jnp.dot(q, k.T) * sm_scale  # (block_q, block_kv)
        # Add relative position bias
        rel_slice = rel_ref[h, q_start:q_end, kv_start:kv_end]
        scores = scores + rel_slice
        # Apply causal mask locally
        q_pos = jnp.arange(q_start, q_end)[:, None]
        kv_pos = jnp.arange(kv_start, kv_end)[None, :]
        mask = q_pos >= kv_pos
        scores = jnp.where(mask, scores, -1e30)
        # Write to scratch
        score_ref[:, kv_start:kv_end] = scores

    # Softmax over key dimension
    max_score = jnp.max(score_ref, axis=-1, keepdims=True)
    exp_scores = jnp.exp(score_ref - max_score)
    sum_exp = jnp.sum(exp_scores, axis=-1, keepdims=True)
    attn = exp_scores / sum_exp

    # Second pass: accumulate output
    out_acc = jnp.zeros((block_q, D), dtype=jnp.float32)
    for kv_idx in range(num_kv_blocks):
        kv_start = kv_idx * block_kv
        kv_end = kv_start + block_kv
        v = v_ref[b, h, kv_start:kv_end, :]
        attn_block = attn[:, kv_start:kv_end]
        out_acc += jnp.dot(attn_block, v)

    out_ref[b, h, q_start:q_end, :] = out_acc.astype(jnp.bfloat16)

def workload(q, k, v, rel_pos_bias):
    out_shape = jax.ShapeDtypeStruct(q.shape, q.dtype)
    return pl.pallas_call(
        flex_kernel,
        out_shape=out_shape,
        grid=(B * H, num_q_blocks),
        in_specs=(
            pl.no_block_spec,
            pl.no_block_spec,
            pl.no_block_spec,
            pl.no_block_spec,
        ),
        out_specs=pl.no_block_spec,
        scratch_shapes=[pltpu.VMEM((block_q, S), jnp.float32)],
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel", "parallel")),
    )(q, k, v, rel_pos_bias)
