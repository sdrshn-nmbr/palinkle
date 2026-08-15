import jax
import jax.numpy as jnp
import jax.pallas as pl
import jax.pallas as plp
import jax.interpreters.pallas as pallas
from jax.interpreters import pallas as pl
from jax.pallas import lib as pllib
import jax.numpy as jnp

def workload(x, gate_kernel, up_kernel, down_kernel):
    """SwiGLU MLP kernel for Llama 3.1 70B.
    
    Computes: output = (SiLU(x @ gate_kernel) * (x @ up_kernel)) @ down_kernel
    """
    # Define block sizes for TPU matmul
    # TPU block dimensions need multiples of 8 for bf16 and 128-element tiling
    block_size = 128
    
    def swiglu_mlp_kernel(ref_x, ref_gate_kernel, ref_up_kernel, ref_down_kernel, ref_out):
        # Compute x @ gate_kernel -> gate
        gate = jnp.dot(ref_x, ref_gate_kernel)
        # Apply SiLU activation: silu(x) = x * sigmoid(x)
        gate = jax.nn.silu(gate)
        
        # Compute x @ up_kernel -> up
        up = jnp.dot(ref_x, ref_up_kernel)
        
        # Element-wise multiply gate and up
        intermediate = gate * up
        
        # Compute (gate * up) @ down_kernel -> output
        output = jnp.dot(intermediate, ref_down_kernel)
        
        ref_out[...] = output
    
    # Output shape
    out_shape = jax.ShapeDtypeStruct(x.shape, x.dtype)
    
    # Grid specification - we process the entire input at once
    # The grid is determined by the output shape
    grid = (1,)
    
    return pl.pallas_call(
        swiglu_mlp_kernel,
        out_shape=out_shape,
        grid=grid,
        in_specs=(
            pl.BlockSpec(x.shape, lambda: (0,)),
            pl.BlockSpec(gate_kernel.shape, lambda: (0,)),
            pl.BlockSpec(up_kernel.shape, lambda: (0,)),
            pl.BlockSpec(down_kernel.shape, lambda: (0,)),
        ),
        out_specs=pl.BlockSpec(x.shape, lambda: (0,)),
        compiler_params=plp.CompilerParams(
            dimension_semantics=("parallel",)
        ),
    )(x, gate_kernel, up_kernel, down_kernel)
