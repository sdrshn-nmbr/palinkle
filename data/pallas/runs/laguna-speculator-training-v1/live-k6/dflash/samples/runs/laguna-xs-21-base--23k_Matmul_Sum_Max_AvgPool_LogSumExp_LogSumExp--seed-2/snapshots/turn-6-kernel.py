import jax
import jax.numpy as jnp
import jax.pallas as pl
import jax.pallas as plp
import jax.interpreters.pallas as pallas
from jax.interpreters import pallas as pl
import jax.numpy as jnp

def workload(x, weight, bias):
    """
    Matmul + Sum + Max + AvgPool + LogSumExp + LogSumExp
    """
    batch_size = x.shape[0]  # 4096
    in_features = x.shape[1]  # 8192
    out_features = weight.shape[1]  # 8192
    
    block_size = 128  # Multiple of 8 for bf16
    
    def kernel(matmul_ref, sum_ref, max_ref, mean_ref, lse1_ref, lse2_ref, out_ref, *args):
        # matmul_ref: (batch_size, out_features) - result of x @ weight + bias
        # We need to compute sum, max, mean, logsumexp along axis 1
        
        # Read the matmul result
        # For simplicity, we'll process the entire output in one block
        # Since output is (4096, 1), we can use a simple grid
        
        # Actually, let's think about this differently
        # The output shape is (4096, 1), so we have 4096 rows
        # Each row needs to go through sum -> max -> mean -> lse -> lse
        
        # Let's use a simpler approach: process each row independently
        row_idx = pl.program_id(0)
        
        if row_idx < batch_size:
            # Read the entire row from matmul result
            row = matmul_ref[row_idx, :]
            
            # Sum along axis 1 (the entire row becomes a scalar, but keepdims)
            s = jnp.sum(row)
            
            # Max along axis 1
            m = jnp.max(row)
            
            # Mean along axis 1
            avg = jnp.mean(row)
            
            # LogSumExp along axis 1
            lse1 = jnp.logsumexp(row, axis=0)
            
            # LogSumExp again
            lse2 = jnp.logsumexp(lse1)
            
            # The output should be the final result
            # Based on the AST, we need to chain: matmul -> sum -> max -> mean -> lse -> lse
            # But looking at the AST more carefully, each operation overwrites x
            # So the final output is the result of the second logsumexp
            
            # Wait, let me re-read the AST...
            # Each Assign updates x, so:
            # x = matmul + bias
            # x = sum(x)
            # x = max(x)
            # x = mean(x)
            # x = logsumexp(x)
            # x = logsumexp(x)
            # return x
            
            # So the operations are chained!
            x_val = jnp.sum(row)
            x_val = jnp.max(x_val)  # This doesn't make sense for a scalar
            # Actually, max of a scalar is the scalar itself
            x_val = jnp.mean(x_val)  # Same issue
            # Hmm, this is confusing. Let me think again.
            
            # Actually, looking at the AST again:
            # sum(x, axis=1, keepdims=True) - this keeps the dimension
            # max(x, axis=1, keepdims=True) - this also keeps the dimension
            # mean(x, axis=1, keepdims=True) - same
            # logsumexp(x, axis=1, keepdims=True) - same
            
            # So after sum, we have shape (4096, 1)
            # After max, we still have shape (4096, 1)
            # etc.
            
            # But wait, if we sum along axis 1 of a (4096, 8192) tensor,
            # we get (4096, 1). Then max of (4096, 1) along axis 1 gives (4096, 1).
            # Same for mean and logsumexp.
            
            # So the operations are:
            # x = (x @ weight + bias)  # shape (4096, 8192)
            # x = sum(x, axis=1, keepdims=True)  # shape (4096, 1)
            # x = max(x, axis=1, keepdims=True)  # shape (4096, 1)
            # x = mean(x, axis=1, keepdims=True)  # shape (4096, 1)
            # x = logsumexp(x, axis=1, keepdims=True)  # shape (4096, 1)
            # x = logsumexp(x, axis=1, keepdims=True)  # shape (4096, 1)
            
            # For a scalar (or (N, 1) tensor), max/mean of a single element is that element
            # So max and mean don't change the value, just the shape
            
            # Let me implement this properly
            pass
    
    # Actually, let me reconsider the approach
    # The simplest way is to use pl.pallas_call with a kernel that processes
    # the entire computation
    
    block = 128
    
    def compute_kernel(x_ref, weight_ref, bias_ref, out_ref):
        # x_ref: (batch_size, in_features)
        # weight_ref: (in_features, out_features)
        # bias_ref: (out_features,)
        # out_ref: (batch_size, 1)
        
        # For each batch element, compute:
        # 1. matmul + bias
        # 2. sum along axis 1
        # 3. max along axis 1
        # 4. mean along axis 1
        # 5. logsumexp along axis 1
        # 6. logsumexp along axis 1
        
        batch_idx = pl.program_id(0)
        
        if batch_idx < x_ref.shape[0]:
            # Read input row
            x_row = x_ref[batch_idx, :]  # (in_features,)
            
            # Compute matmul + bias
            # We need to accumulate in float32 for numerical stability
            result = jnp.zeros(x_ref.shape[1], dtype=jnp.float32)
            for k in range(weight_ref.shape[0]):
                result = result + x_row[k] * weight_ref[k, :]
            result = result + bias_ref
            
            # Convert back to bf16 for subsequent operations
            result = result.astype(jnp.bfloat16)
            
            # Sum along axis 0 (since we have a 1D array now)
            s = jnp.sum(result)
            
            # Max
            m = jnp.max(s)
            
            # Mean
            avg = jnp.mean(m)
            
            # LogSumExp
            lse1 = jnp.logsumexp(avg)
            
            # LogSumExp again
            lse2 = jnp.logsumexp(lse1)
            
            out_ref[batch_idx, 0] = lse2.astype(jnp.bfloat16)
    
    return pl.pallas_call(
        compute_kernel,
        out_shape=jax.ShapeDtypeStruct((batch_size, 1), jnp.bfloat16),
        grid=(batch_size,),
        in_specs=(
            pl.BlockSpec((batch_size, in_features), lambda _: (0, 0)),
            pl.BlockSpec((in_features, out_features), lambda _: (0, 0)),
            pl.BlockSpec((out_features,), lambda _: (0,)),
        ),
        out_specs=pl.BlockSpec((batch_size, 1), lambda i: (i, 0)),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel",)
        ),
    )(x, weight, bias)
