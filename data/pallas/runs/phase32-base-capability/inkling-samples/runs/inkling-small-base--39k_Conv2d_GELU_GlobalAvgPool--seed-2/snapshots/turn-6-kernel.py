import jax
import jax.numpy as jnp
import jax.experimental.pallas as pl

def workload(x, weight, bias):
    x_t = jnp.transpose(x, (0, 2, 3, 1))
    w_t = jnp.transpose(weight, (2, 3, 1, 0))
    
    def kernel(x_ref, w_ref, b_ref, out_ref):
        b = pl.program_id(0)
        c = pl.program_id(1)
        accum = jnp.zeros((254, 254), dtype=jnp.float32)
        for dh in range(3):
            for dw in range(3):
                for ic in range(8):
                    x_slice = x_ref[b, dh:dh+254, dw:dw+254, ic]
                    w_val = w_ref[dh, dw, ic, c]
                    accum += x_slice.astype(jnp.float32) * w_val.astype(jnp.float32)
        accum += b_ref[c].astype(jnp.float32)
        gelu_val = accum * 0.5 * (1.0 + jnp.erf(accum / jnp.sqrt(2.0)))
        mean_val = jnp.mean(gelu_val)
        out_ref[b, c] = mean_val.astype(jnp.bfloat16)
    
    out_shape = jax.ShapeDtypeStruct((128, 64), jnp.bfloat16)
    
    return pl.pallas_call(
        kernel,
        out_shape=out_shape,
        grid=(128, 64),
        in_specs=(
            pl.BlockSpec((1, 256, 256, 8), lambda b, c: (b, 0, 0, 0)),
            pl.BlockSpec((3, 3, 8, 1), lambda b, c: (0, 0, 0, c)),
            pl.BlockSpec((64,), lambda b, c: (0,)),
        ),
        out_specs=pl.BlockSpec((1, 1), lambda b, c: (b, c)),
    )(x_t, w_t, bias)
