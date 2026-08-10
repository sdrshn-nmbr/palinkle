from jax.experimental.pallas.ops.tpu.flash_attention import flash_attention


def workload(query, key, value):
    return flash_attention(
        query,
        key,
        value,
        causal=True,
        sm_scale=query.shape[-1] ** -0.5,
    )
