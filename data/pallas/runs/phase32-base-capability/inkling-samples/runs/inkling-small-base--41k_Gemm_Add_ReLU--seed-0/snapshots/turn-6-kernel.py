import jax
import jax.numpy as jnp
import jax.experimental.pallas as pl
import jax.experimental.pallas.tpu as pltpu

def workload(x, weight, bias):
    def kernel(x_ref, w_ref, b_ref, out_ref, scratch_ref):
        # Accumulate matmul in float32 VMEM scratch
        # x_ref: [4096, 8192] bf16, w_ref: [8192, 8192] bf16
        x_f32 = x_ref[...].astype(jnp.float32)
        w_f32 = w_ref[...].astype(jnp.float32)
        # Compute dot product
        scratch_ref[...] = jnp.dot(x_f32, w_f32)
        # Add bias (broadcast over first dim)
        b_f32 = b_ref[...].astype(jnp.float32)
        scratch_ref[...] = scratch_ref[...] + b_f32
        # ReLU
        scratch_ref[...] = jnp.maximum(scratch_ref[...], 0.0)
        # Write output as bf16
        out_ref[...] = scratch_ref[...].astype(jnp.bfloat16)

    out_shape = jax.ShapeDtypeStruct(x.shape, x.dtype)
    scratch_shape = pltpu.VMEM(x.shape, jnp.float32)
    return pl.pallas_call(
        kernel,
        out_shape=out_shape,
        grid=(1, 1),
        in_specs=(
            pl.BlockSpec(x.shape, lambda i, j: (0, 0)),
            pl.BlockSpec(weight.shape, lambda i, j: (0, 0)),
            pl.BlockSpec(bias.shape, lambda i, j: (0,)),
        ),
        out_specs=pl.BlockSpec(x.shape, lambda i, j: (0, 0)),
        scratch_shapes=(scratch_shape,),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel",)),
    )(x, weight, bias)
