import jax
import jax.numpy as jnp
import jax.lax as lax
from jax import random
import jax.experimental.pallas as pl
import jax.experimental.pallas.tpu as pltpu

def workload(pair_act, mask, left_proj_w, right_proj_w, left_gate_w, right_gate_w, center_scale, out_proj_w, out_gate_w):
    def kernel(pair_act_ref, mask_ref, left_proj_w_ref, right_proj_w_ref, left_gate_w_ref, right_gate_w_ref, center_scale_ref, out_proj_w_ref, out_gate_w_ref, out_ref):
        pair_act = pair_act_ref[...]
        mask = mask_ref[...]
        left_proj_w = left_proj_w_ref[...]
        right_proj_w = right_proj_w_ref[...]
        left_gate_w = left_gate_w_ref[...]
        right_gate_w = right_gate_w_ref[...]
        center_scale = center_scale_ref[...]
        out_proj_w = out_proj_w_ref[...]
        out_gate_w = out_gate_w_ref[...]

        act = pair_act * mask
        left_proj = jnp.dot(act, left_proj_w)
        right_proj = jnp.dot(act, right_proj_w)
        left_gate = jax.nn.sigmoid(jnp.dot(act, left_gate_w))
        right_gate = jax.nn.sigmoid(jnp.dot(act, right_gate_w))
        left_proj = left_proj * left_gate
        right_proj = right_proj * right_gate
        result = jnp.einsum("ikc,jkc->ijc", left_proj, right_proj)
        eps = 1e-06
        rms = jnp.sqrt(jnp.mean(result * result, axis=-1, keepdims=True) + eps)
        result = (result / rms) * center_scale
        output = jnp.dot(result, out_proj_w)
        gate = jax.nn.sigmoid(jnp.dot(pair_act, out_gate_w))
        out_ref[...] = output * gate

    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct(pair_act.shape, pair_act.dtype),
        grid=(),
        in_specs=[pl.no_block_spec] * 9,
        out_specs=pl.no_block_spec,
    )(pair_act, mask, left_proj_w, right_proj_w, left_gate_w, right_gate_w, center_scale, out_proj_w, out_gate_w)
