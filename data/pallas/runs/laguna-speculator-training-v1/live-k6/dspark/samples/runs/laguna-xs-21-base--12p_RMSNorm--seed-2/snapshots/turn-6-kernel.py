import jax
import jax.numpy as jnp
import jax.pallas as pl
import jax.pallas.lib as pllib
from jax.interpreters import pallas as pallas
import jax.lax as lax

def workload(x, scale):
    batch, seq_len, emb_dim = x.shape
    
    def rmsnorm_kernel(x_ref, scale_ref, out_ref):
        # x_ref has shape (batch_block, seq_block, emb_block)
        # scale_ref has shape (emb_block,)
        # out_ref has shape (batch_block, seq_block, emb_block)
        
        # Convert to float32 for numerical stability
        x_f32 = x_ref[...].astype(jnp.float32)
        
        # Compute mean of x^2 along the emb_dim axis (axis=-1)
        # keepdims=True to broadcast back
        mean2 = jnp.mean(x_f32 ** 2, axis=-1, keepdims=True)
        
        # Compute rsqrt(mean2 + epsilon)
        eps = 1e-5
        inv_std = lax.rsqrt(mean2 + eps)
        
        # Normalize: x * rsqrt(mean(x^2) + eps)
        normed = x_f32 * inv_std
        
        # Convert back to bfloat16
        normed = normed.astype(x_ref.dtype)
        
        # Apply scale
        out_ref[...] = normed * scale_ref[None, None, :]
    
    # Use block size of 128 for emb_dim (multiple of 128 for TPU vectorization)
    block_size = 128
    
    return pl.pallas_call(
        rmsnorm_kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        grid=(batch, seq_len, emb_dim // block_size),
        in_specs=(
            pl.BlockSpec((1, 1, block_size), lambda b, s, e: (b, s, e * block_size)),
            pl.BlockSpec((block_size,), lambda b, s, e: (e * block_size,)),
        ),
        out_specs=pl.BlockSpec((1, 1, block_size), lambda b, s, e: (b, s, e * block_size)),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel", "parallel")
        ),
    )(x, scale)
