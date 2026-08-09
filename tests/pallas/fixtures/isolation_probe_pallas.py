import os
import socket
from pathlib import Path

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl


protected_paths = (
    Path("/tmp/opjax-phase2-submission/input/isolation-secret"),
    Path("/home/sudarshan/opjax-phase2-final/input/isolation-secret"),
)
if any(os.access(path, os.R_OK) for path in protected_paths):
    raise RuntimeError("PROTECTED_HOST_PATH_READABLE")
try:
    socket.create_connection(("169.254.169.254", 80), timeout=1)
except OSError:
    pass
else:
    raise RuntimeError("METADATA_NETWORK_REACHABLE")


def _kernel(x_ref, weight_ref, out_ref):
    values = x_ref[...].astype(jnp.float32)
    normalized = values * jax.lax.rsqrt(
        jnp.mean(jnp.square(values), axis=-1, keepdims=True) + 1e-5
    )
    out_ref[...] = normalized.astype(x_ref.dtype) * weight_ref[...]


def workload(x, weight):
    x_spec = pl.BlockSpec((4, 128, 512), lambda _: (0, 0, 0))
    return pl.pallas_call(
        _kernel,
        out_shape=jax.ShapeDtypeStruct((4, 128, 512), jnp.float32),
        grid=(1,),
        in_specs=(x_spec, pl.BlockSpec((512,), lambda _: (0,))),
        out_specs=x_spec,
    )(x, weight)
