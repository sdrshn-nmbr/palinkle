import jax
import jax.numpy as jnp
import jax.scipy.special
from jax import lax
import jax.experimental.pallas as pl
import jax.experimental.pallas.tpu as pltpu

def workload(x, w1, b1, w2, b2):
    batch_size = x.shape[0]
    batch_block = 128
    
    def kernel(x_ref, w1_ref, b1_ref, w2_ref, b2_ref, out_ref):
        # Load block of x
        x_block = x_ref[...].astype(jnp.float32)
        w1_full = w1_ref[...].astype(jnp.float32)
        b1_full = b1_ref[...].astype(jnp.float32)
        w2_full = w2_ref[...].astype(jnp.float32)
        b2_full = b2_ref[...].astype(jnp.float32)
        
        # First gemm + bias
        h = jnp.dot(x_block, w1_full.T) + b1_full
        h = jax.nn.sigmoid(h)
        
        # Second gemm + bias
        y = jnp.dot(h, w2_full.T) + b2_full
        
        # Logsumexp over axis=1
        out_ref[...] = jax.scipy.special.logsumexp(y, axis=1).astype(jnp.bfloat16)
    
    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((batch_size,), jnp.bfloat16),
        grid=(batch_size // batch_block,),
        in_specs=(
            pl.BlockSpec((batch_block, x.shape[1]), lambda i: (i, 0)),
            pl.no_block_spec,
            pl.no_block_spec,
            pl.no_block_spec,
            pl.no_block_spec,
        ),
        out_specs=pl.BlockSpec((batch_block,), lambda i: (i,)),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel",)),
    )(x, w1, b1, w2, b2)
