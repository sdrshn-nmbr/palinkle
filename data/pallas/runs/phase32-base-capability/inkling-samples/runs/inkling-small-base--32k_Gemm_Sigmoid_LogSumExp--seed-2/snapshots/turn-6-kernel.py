import jax
import jax.numpy as jnp
import jax.lax as lax
from jax import scipy
import jax.experimental.pallas as pl
import jax.experimental.pallas.tpu as pltpu

def workload(x, w1, b1, w2, b2):
    batch_size = x.shape[0]
    block_batch = 8
    
    def kernel(x_ref, w1_ref, b1_ref, w2_ref, b2_ref, out_ref):
        # Load block of x
        x_block = x_ref[...]  # (block_batch, 2048)
        
        # Load full weights (passed with no_block_spec)
        w1_full = w1_ref[...]  # (4096, 2048)
        b1_full = b1_ref[...]  # (4096,)
        w2_full = w2_ref[...]  # (1024, 4096)
        b2_full = b2_ref[...]  # (1024,)
        
        # First gemm: x @ w1.T + b1
        # x_block: (B, 2048), w1_full.T: (2048, 4096)
        y = jnp.dot(x_block, w1_full.T) + b1_full  # (B, 4096)
        
        # Sigmoid
        y = jax.nn.sigmoid(y)
        
        # Second gemm: y @ w2.T + b2
        y = jnp.dot(y, w2_full.T) + b2_full  # (B, 1024)
        
        # LogSumExp over axis=1
        y = scipy.special.logsumexp(y, axis=1)  # (B,)
        
        out_ref[...] = y
    
    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((batch_size,), jnp.bfloat16),
        grid=(batch_size // block_batch,),
        in_specs=(
            pl.BlockSpec((block_batch, 2048), lambda i: (i * block_batch, 0)),
            pl.no_block_spec,
            pl.no_block_spec,
            pl.no_block_spec,
            pl.no_block_spec,
        ),
        out_specs=pl.BlockSpec((block_batch,), lambda i: (i * block_batch,)),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel",)),
    )(x, w1, b1, w2, b2)
