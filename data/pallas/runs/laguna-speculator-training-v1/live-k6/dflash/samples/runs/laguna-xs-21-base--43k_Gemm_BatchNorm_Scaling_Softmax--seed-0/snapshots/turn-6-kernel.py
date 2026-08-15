import jax
import jax.numpy as jnp
import jax.pallas as pl
import jax.pallas as plp
import jax.interpreters.pallas as pj
import functools

def workload(x, weight, bias, bn_scale, bn_bias, bn_mean, bn_var, scale):
    """
    GEMM + BatchNorm + Scaling + Softmax kernel.
    
    Operations:
    1. x = x @ weight + bias
    2. x_normalized = (x - bn_mean) / sqrt(bn_var + eps)
    3. x = bn_scale * x_normalized + bn_bias
    4. x = scale * x
    5. x = softmax(x, axis=1)
    """
    eps = 1e-5
    
    # Output shape
    out_shape = jax.ShapeDtypeStruct(x.shape, x.dtype)
    
    # Block size for TPU - need multiples of 8 for bf16
    block_size = 128
    
    # Grid dimensions: (batch_blocks, feature_blocks)
    # x is (4096, 8192), so we need 4096/128 = 32 blocks in batch dim
    # and 8192/128 = 64 blocks in feature dim
    grid = (x.shape[0] // block_size, x.shape[1] // block_size)
    
    def kernel(ref_out, x_ref, weight_ref, bias_ref, bn_scale_ref, bn_bias_ref, 
               bn_mean_ref, bn_var_ref, scale_ref):
        # Get program IDs
        batch_block = pl.program_id(0)
        feat_block = pl.program_id(1)
        
        # Compute tile offsets
        batch_start = batch_block * block_size
        feat_start = feat_block * block_size
        
        # Initialize output tile
        ref_out[...] = 0.0
        
        # GEMM: compute x @ weight + bias
        # For each element in the output tile
        for i in pl.range(block_size):
            for j in pl.range(block_size):
                # Compute dot product for this element
                # x[batch_start + i, :] @ weight[:, feat_start + j]
                acc = 0.0
                for k in pl.range(x.shape[1]):
                    x_val = x_ref[batch_start + i, k]
                    w_val = weight_ref[k, feat_start + j]
                    acc += float(x_val) * float(w_val)
                
                # Add bias
                acc += float(bias_ref[feat_start + j])
                
                # Store intermediate result for batch norm
                # We'll do batch norm and softmax in a second pass
                # For now, store in ref_out
                ref_out[batch_start + i, feat_start + j] = acc
    
    # First pass: GEMM
    def gemm_kernel(ref_out, x_ref, weight_ref, bias_ref):
        batch_block = pl.program_id(0)
        feat_block = pl.program_id(1)
        
        batch_start = batch_block * block_size
        feat_start = feat_block * block_size
        
        # Compute dot product for this tile
        acc = 0.0
        for k in pl.range(x.shape[1]):
            x_val = x_ref[batch_start, k]
            w_val = weight_ref[k, feat_start]
            acc += float(x_val) * float(w_val)
        
        acc += float(bias_ref[feat_start])
        ref_out[...] = acc
    
    # Simplified approach: use JAX operations inside Pallas kernel
    # The kernel will process the entire matrix at once
    
    def full_kernel(ref_out, x_ref, weight_ref, bias_ref, bn_scale_ref, bn_bias_ref,
                    bn_mean_ref, bn_var_ref, scale_ref):
        # Get the full input arrays
        x_val = x_ref[...]
        weight_val = weight_ref[...]
        bias_val = bias_ref[...]
        bn_scale_val = bn_scale_ref[...]
        bn_bias_val = bn_bias_ref[...]
        bn_mean_val = bn_mean_ref[...]
        bn_var_val = bn_var_ref[...]
        scale_val = scale_ref[...]
        
        # GEMM: x @ weight + bias
        x_val = jnp.dot(x_val, weight_val) + bias_val
        
        # BatchNorm
        x_normalized = (x_val - bn_mean_val) / jnp.sqrt(bn_var_val + eps)
        x_val = bn_scale_val * x_normalized + bn_bias_val
        
        # Scaling
        x_val = scale_val * x_val
        
        # Softmax along axis 1
        x_val = jax.nn.softmax(x_val, axis=1)
        
        ref_out[...] = x_val
    
    return pl.pallas_call(
        full_kernel,
        out_shape=out_shape,
        grid=grid,
        in_specs=(
            pl.BlockSpec((x.shape[0], x.shape[1]), lambda: (0, 0)),
            pl.BlockSpec((weight.shape[0], weight.shape[1]), lambda: (0, 0)),
            pl.BlockSpec((bias.shape[0],), lambda: (0,)),
            pl.BlockSpec((bn_scale.shape[0],), lambda: (0,)),
            pl.BlockSpec((bn_bias.shape[0],), lambda: (0,)),
            pl.BlockSpec((bn_mean.shape[0],), lambda: (0,)),
            pl.BlockSpec((bn_var.shape[0],), lambda: (0,)),
            pl.BlockSpec((scale.shape[0],), lambda: (0,)),
        ),
        out_specs=pl.BlockSpec((x.shape[0], x.shape[1]), lambda: (0, 0)),
        compiler_params=jax.pallas TPUCompilerParams(
            dimension_semantics=("parallel", "parallel")
        ),
    )(x, weight, bias, bn_scale, bn_bias, bn_mean, bn_var, scale)
