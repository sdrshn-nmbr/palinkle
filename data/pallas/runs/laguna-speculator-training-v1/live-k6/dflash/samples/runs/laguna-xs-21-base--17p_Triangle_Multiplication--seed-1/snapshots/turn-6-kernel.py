import jax
import jax.numpy as jnp
import jax.pallas as pl
import jax.pallas.tpu as pltpu

def workload(pair_act, mask, left_proj_w, right_proj_w, left_gate_w, right_gate_w, center_scale, out_proj_w, out_gate_w):
    """Triangle Multiplicative Update (Outgoing) kernel for AlphaFold2."""
    
    # Define block sizes (multiples of 8 for bf16 efficiency on TPU)
    block_i = 128
    block_j = 128
    block_c = 128
    
    # Grid dimensions
    N = 1536  # pair_act.shape[0]
    C = 128   # pair_act.shape[2]
    
    grid = (N // block_i, N // block_j, C // block_c)
    
    def triangle_mult_kernel(
        pair_act_ref,
        mask_ref,
        left_proj_w_ref,
        right_proj_w_ref,
        left_gate_w_ref,
        right_gate_w_ref,
        center_scale_ref,
        out_proj_w_ref,
        out_gate_w_ref,
        out_ref,
    ):
        # Get block indices
        i_block = pl.program_id(0)
        j_block = pl.program_id(1)
        c_block = pl.program_id(2)
        
        # Compute tile ranges
        i_start = i_block * block_i
        i_end = min(i_start + block_i, N)
        j_start = j_block * block_j
        j_end = min(j_start + block_j, N)
        c_start = c_block * block_c
        c_end = min(c_start + block_c, C)
        
        # Load tiles of pair_act and mask
        act_tile = pair_act_ref[i_start:i_end, :, :] * mask_ref[i_start:i_end, :, :]
        
        # Load weights
        left_proj_w = left_proj_w_ref[:]
        right_proj_w = right_proj_w_ref[:]
        left_gate_w = left_gate_w_ref[:]
        right_gate_w = right_gate_w_ref[:]
        center_scale = center_scale_ref[:]
        out_proj_w = out_proj_w_ref[:]
        out_gate_w = out_gate_w_ref[:]
        
        # Compute projections
        left_proj = jnp.dot(act_tile, left_proj_w)
        right_proj = jnp.dot(act_tile, right_proj_w)
        
        # Compute gates
        left_gate = jax.nn.sigmoid(jnp.dot(act_tile, left_gate_w))
        right_gate = jax.nn.sigmoid(jnp.dot(act_tile, right_gate_w))
        
        # Apply gates
        left_proj = left_proj * left_gate
        right_proj = right_proj * right_gate
        
        # Compute einsum: "ikc,jkc->ijc"
        # This contracts over the second residue index (k dimension)
        result = jnp.einsum("ikc,jkc->ijc", left_proj, right_proj)
        
        # RMS normalization
        eps = 1e-6
        rms = jnp.sqrt(jnp.mean(result * result, axis=-1, keepdims=True) + eps)
        result = (result / rms) * center_scale
        
        # Output projection
        output = jnp.dot(result, out_proj_w)
        
        # Output gate
        gate = jax.nn.sigmoid(jnp.dot(pair_act_ref[i_start:i_end, :, :], out_gate_w))
        
        # Apply gate
        output = output * gate
        
        # Write output tile
        out_ref[i_start:i_end, j_start:j_end, c_start:c_end] = output[:, :, c_start:c_end]
    
    # Define output shape
    out_shape = jax.ShapeDtypeStruct(pair_act.shape, pair_act.dtype)
    
    # Define input specs
    in_specs = (
        pl.BlockSpec((block_i, N, C), lambda i, j, k: (i * block_i, 0, 0)),
        pl.BlockSpec((block_i, N, 1), lambda i, j, k: (i * block_i, 0, 0)),
        pl.BlockSpec((C, C), lambda i, j, k: (0, 0)),
        pl.BlockSpec((C, C), lambda i, j, k: (0, 0)),
        pl.BlockSpec((C, C), lambda i, j, k: (0, 0)),
        pl.BlockSpec((C, C), lambda i, j, k: (0, 0)),
        pl.BlockSpec((C,), lambda i, j, k: (0,)),
        pl.BlockSpec((C, C), lambda i, j, k: (0, 0)),
        pl.BlockSpec((C, C), lambda i, j, k: (0, 0)),
    )
    
    # Define output spec
    out_specs = pl.BlockSpec((block_i, block_j, block_c), lambda i, j, k: (i * block_i, j * block_j, k * block_c))
    
    return pl.pallas_call(
        triangle_mult_kernel,
        out_shape=out_shape,
        grid=grid,
        in_specs=in_specs,
        out_specs=out_specs,
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel", "parallel")
        ),
    )(pair_act, mask, left_proj_w, right_proj_w, left_gate_w, right_gate_w, center_scale, out_proj_w, out_gate_w)
