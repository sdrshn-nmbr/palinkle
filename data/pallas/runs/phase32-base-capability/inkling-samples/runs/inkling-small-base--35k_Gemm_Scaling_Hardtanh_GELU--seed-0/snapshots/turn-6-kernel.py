import jax
import jax.numpy as jnp
import jax.lax as lax
from jax import random
import jax.experimental.pallas as pl
import jax.experimental.pallas.tpu as pltpu

def workload(x, weight, bias):
    batch_size, in_features = x.shape
    out_features = weight.shape[1]
    
    block_b = 128
    block_f = 128
    
    grid_b = batch_size // block_b
    grid_f = out_features // block_f
    
    def kernel(x_ref, w_ref, b_ref, out_ref):
        # Load inputs as float32 for accumulation
        x_local = x_ref[...].astype(jnp.float32)
        w_local = w_ref[...].astype(jnp.float32)
        b_local = b_ref[...].astype(jnp.float32)
        
        # Matmul + bias
        out_local = jnp.dot(x_local, w_local) + b_local
        
        # Scaling by 0.5
        out_local = out_local * 0.5
        
        # Hardtanh clip to [-2, 2]
        out_local = jnp.clip(out_local, -2.0, 2.0)
        
        # GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        # But the contract says: x = x * 0.5 * (1.0 + tanh(...))
        # Wait, looking at AST: left = x * 0.5, right = (1 + tanh(...))
        # So it's x * 0.5 * (1 + tanh(...))
        # Actually the AST shows: BinOp(left=BinOp(left=x, op=Mult, right=0.5), op=Mult, right=BinOp(...))
        # So yes: (x * 0.5) * (1 + tanh(...))
        
        # Compute GELU approximation
        sqrt_2_over_pi = jnp.sqrt(2.0 / jnp.pi)
        gelu_inner = sqrt_2_over_pi * (out_local + 0.044715 * (out_local ** 3))
        gelu_tanh = jnp.tanh(gelu_inner)
        gelu_factor = 1.0 + gelu_tanh
        out_local = out_local * 0.5 * gelu_factor
        
        out_ref[...] = out_local.astype(jnp.bfloat16)
    
    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, jnp.bfloat16),
        grid=(grid_b, grid_f),
        in_specs=(
            pl.BlockSpec((block_b, in_features), lambda i, j: (i * block_b, 0)),
            pl.BlockSpec((in_features, block_f), lambda i, j: (0, j * block_f)),
            pl.BlockSpec((block_f,), lambda i, j: (j * block_f,)),
        ),
        out_specs=pl.BlockSpec((block_b, block_f), lambda i, j: (i * block_b, j * block_f)),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel", "parallel")),
    )(x, weight, bias)
