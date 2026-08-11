import jax
import jax.numpy as jnp
import jax.experimental.pallas as pl
import jax.experimental.pallas.tpu as pltpu

def workload(x, weight, bias, bn_scale, bn_bias, bn_mean, bn_var, scale):
    def kernel(x_ref, weight_ref, bias_ref, bn_scale_ref, bn_bias_ref, bn_mean_ref, bn_var_ref, scale_ref, out_ref, scratch_ref):
        # x_ref is already (128, 8192) block
        x_block = x_ref[...]
        scratch_ref[...] = jnp.zeros((128, 8192), dtype=jnp.float32)
        chunk = 256
        for k in range(0, 8192, chunk):
            x_chunk = x_block[:, k:k + chunk]
            w_chunk = weight_ref[k:k + chunk, :]
            scratch_ref[...] += jnp.dot(x_chunk, w_chunk)
        scratch_ref[...] += bias_ref[:]
        bn_eps = 1e-05
        x_normalized = (scratch_ref[...] - bn_mean_ref[:]) / jnp.sqrt(bn_var_ref[:] + bn_eps)
        scratch_ref[...] = bn_scale_ref[:] * x_normalized + bn_bias_ref[:]
        scratch_ref[...] *= scale_ref[0]
        scratch_ref[...] = jax.nn.softmax(scratch_ref[...], axis=1)
        out_ref[...] = scratch_ref[...].astype(jnp.bfloat16)

    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        grid=(4096 // 128,),
        in_specs=(
            pl.BlockSpec((128, 8192), lambda i: (i * 128, 0)),
            pl.no_block_spec,
            pl.no_block_spec,
            pl.no_block_spec,
            pl.no_block_spec,
            pl.no_block_spec,
            pl.no_block_spec,
            pl.no_block_spec,
        ),
        out_specs=pl.BlockSpec((128, 8192), lambda i: (i * 128, 0)),
        scratch_shapes=(pltpu.VMEM((128, 8192), jnp.float32),),
    )(x, weight, bias, bn_scale, bn_bias, bn_mean, bn_var, scale)
