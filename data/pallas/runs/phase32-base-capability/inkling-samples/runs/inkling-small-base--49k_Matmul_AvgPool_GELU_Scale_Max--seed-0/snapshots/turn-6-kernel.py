import jax
import jax.numpy as jnp
import jax.lax as lax
from jax import random
import jax.experimental.pallas as pl
import jax.experimental.pallas.tpu as pltpu

# Stage 1: Matmul + Bias
# Grid over batch (4096) and out_features tiles (8192/128 = 64)
# Each kernel computes a (1, 128) output tile

def matmul_bias_kernel(x_ref, w_ref, b_ref, out_ref):
    # x_ref: (1, 128) - actually we need full row for dot
    # Better: load x row in chunks
    pass

# Actually let's use a simpler approach: process full batch rows with full weight tiles
# Given time, let's implement a pipeline with simpler kernels

def workload(x, weight, bias):
    # Stage 1: matmul + bias
    # We'll use a tiled approach with grid over batch and feature tiles
    batch_size, in_features = x.shape
    out_features = weight.shape[1]
    
    # For simplicity, use pallas_call with a kernel that does matmul via jnp.dot
    # on appropriately tiled inputs. Given the large sizes, we'll tile over batch.
    
    # Actually, let's implement with grid=(batch_size,) and block_shape=(1, in_features) for x
    # and (in_features, 128) for weight, computing output tiles of (1, 128)
    
    # To keep it manageable, let's do a single pallas_call that computes the full matmul
    # by having each program handle one batch row and accumulate over weight tiles
    
    def matmul_kernel(x_ref, w_ref, b_ref, out_ref):
        # x_ref shape: (1, 8192) - one batch row
        # w_ref shape: (8192, 128) - weight tile
        # b_ref shape: (128,) - bias tile
        # out_ref shape: (1, 128)
        
        # Load x row
        x_row = x_ref[...]
        # Compute dot product: x_row (1, 8192) @ w_ref (8192, 128) = (1, 128)
        # But x_ref might be tiled differently
        
        # Actually let's just use jnp.dot with the references
        # We assume the block specs align so that x_ref is (1, 8192) and w_ref is (8192, 128)
        # But that requires loading full weight per batch row which is 134MB
        
        # For correctness, let's do accumulation in float32
        acc = jnp.dot(x_ref[...].astype(jnp.float32), w_ref[...].astype(jnp.float32))
        out_ref[...] = (acc + b_ref[...].astype(jnp.float32)).astype(jnp.bfloat16)
    
    # Grid: (4096, 64) for batch and out_feature tiles of 128
    # But we need to handle the tiling properly
    
    # Actually, let's use a simpler grid: (4096,) with block_shape=(1, 8192) for x
    # and for weight we use a scalar prefetch or load in chunks inside kernel
    
    # Given complexity, let's implement as sequential pallas calls with full operations
    # where possible, using block specs that match
    
    # For stage 1, let's try grid=(4096,) with x block=(1, 8192), weight block=(8192, 8192)
    # This loads full weight for each batch row - memory heavy but might work
    
    # Actually, let's just implement the whole pipeline with a single kernel that uses
    # jnp.dot on the full inputs, relying on Pallas to handle tiling automatically
    # when using appropriate block specs.
    
    # Wait - inside a Pallas kernel, jnp.dot operates on the loaded references.
    # If we set block specs to load full matrices, it will work but might be slow.
    
    # Let's try: grid=(), in_specs=no_block_spec, out_specs=no_block_spec
    # Then inside kernel, we load full inputs and compute.
    
    def full_kernel(x_ref, w_ref, b_ref, out_ref):
        # Load full inputs
        x_full = x_ref[...]
        w_full = w_ref[...]
        b_full = b_ref[...]
        
        # Matmul + bias
        y = jnp.dot(x_full, w_full) + b_full
        
        # Expand dims axis=1
        y = jnp.expand_dims(y, axis=1)
        
        # Avg pool: reduce_window with window_dims=(1,1,16), strides=(1,1,16), valid
        # We can implement with lax.reduce_window
        y = lax.reduce_window(
            y,
            init_value=0.0,
            computation=lax.add,
            window_dimensions=(1, 1, 16),
            window_strides=(1, 1, 16),
            padding="VALID"
        )
        y = y / 16.0
        
        # Squeeze axis=1
        y = jnp.squeeze(y, axis=1)
        
        # GELU
        y = jax.nn.gelu(y)
        
        # Scale
        y = y * 2.0
        
        # Max over axis=1
        y = jnp.max(y, axis=1)
        
        out_ref[...] = y.astype(jnp.bfloat16)
    
    # Use grid=() with no block specs - single program handles everything
    # This is valid Pallas and should lower correctly
    out_shape = jax.ShapeDtypeStruct((4096,), jnp.bfloat16)
    
    return pl.pallas_call(
        full_kernel,
        out_shape=out_shape,
        grid=(),
        in_specs=(
            pl.BlockSpec((4096, 8192), lambda: (0, 0)),
            pl.BlockSpec((8192, 8192), lambda: (0, 0)),
            pl.BlockSpec((8192,), lambda: (0,)),
        ),
        out_specs=pl.BlockSpec((4096,), lambda: (0,)),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel",)),
    )(x, weight, bias)
