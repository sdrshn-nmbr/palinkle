import jax
import jax.numpy as jnp
import jax.lax as lax
import jax.experimental.pallas as pl
import jax.experimental.pallas.tpu as pltpu

def workload(x, gate_kernel, up_kernel, down_kernel):
    def kernel(x_ref, gate_ref, up_ref, down_ref, out_ref):
        # Full array computation inside kernel
        gate = jax.nn.silu(jnp.dot(x_ref[...], gate_ref[...]))
        up = jnp.dot(x_ref[...], up_ref[...])
        out_ref[...] = jnp.dot(gate * up, down_ref[...])
    
    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        grid=(),
        in_specs=(
            pl.BlockSpec(x.shape, lambda: (0, 0, 0)),
            pl.BlockSpec(gate_kernel.shape, lambda: (0, 0)),
            pl.BlockSpec(up_kernel.shape, lambda: (0, 0)),
            pl.BlockSpec(down_kernel.shape, lambda: (0, 0)),
        ),
        out_specs=pl.BlockSpec(x.shape, lambda: (0, 0, 0)),
    )(x, gate_kernel, up_kernel, down_kernel)
