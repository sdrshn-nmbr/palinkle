import jax
import jax.numpy as jnp
import jax.pallas as pl
import jax.pallas as plp
import jax.interpreters.pallas as pallas
from jax.interpreters import pallas as pl
import jax.numpy as jnp

def workload(x, w1, b1, w2, b2):
    """
    Compute: logsumexp(sigmoid(x @ w1.T + b1) @ w2.T + b2, axis=1)
    """
    batch_size = x.shape[0]
    hidden_size = w1.shape[0]  # 4096
    output_size = w2.shape[0]  # 1024
    
    # Block size for matmul operations
    block_m = 128
    block_k = 128
    block_n = 128
    
    def kernel(ref_x, ref_w1, ref_b1, ref_w2, ref_b2, ref_out):
        # Step 1: x @ w1.T + b1 -> intermediate shape [batch_size, hidden_size]
        # We need to compute this in the kernel
        
        # For simplicity, we'll use jnp operations inside the kernel
        # The kernel will process the entire computation
        
        # Read inputs
        x_local = ref_x[...]
        w1_local = ref_w1[...]
        b1_local = ref_b1[...]
        w2_local = ref_w2[...]
        b2_local = ref_b2[...]
        
        # Step 1: matmul + bias
        # x @ w1.T: [batch, 2048] @ [2048, 4096] = [batch, 4096]
        hidden = jnp.dot(x_local, w1_local.T) + b1_local
        
        # Step 2: sigmoid
        hidden = jax.nn.sigmoid(hidden)
        
        # Step 3: matmul + bias
        # hidden @ w2.T: [batch, 4096] @ [4096, 1024] = [batch, 1024]
        output = jnp.dot(hidden, w2_local.T) + b2_local
        
        # Step 4: logsumexp along axis 1
        result = jax.scipy.special.logsumexp(output, axis=1)
        
        ref_out[...] = result
    
    # Grid: one block per batch element
    grid = (batch_size,)
    
    # Block specs
    in_specs = (
        pl.BlockSpec((batch_size, x.shape[1]), lambda: (0, 0)),  # x
        pl.BlockSpec((hidden_size, x.shape[1]), lambda: (0, 0)),  # w1
        pl.BlockSpec((hidden_size,), lambda: (0,)),  # b1
        pl.BlockSpec((output_size, hidden_size), lambda: (0, 0)),  # w2
        pl.BlockSpec((output_size,), lambda: (0,)),  # b2
    )
    
    out_specs = pl.BlockSpec((batch_size,), lambda: (0,))
    
    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((batch_size,), x.dtype),
        grid=grid,
        in_specs=in_specs,
        out_specs=out_specs,
        compiler_params=plp.CompilerParams(
            dimension_semantics=("parallel",)
        ),
    )(x, w1, b1, w2, b2)
