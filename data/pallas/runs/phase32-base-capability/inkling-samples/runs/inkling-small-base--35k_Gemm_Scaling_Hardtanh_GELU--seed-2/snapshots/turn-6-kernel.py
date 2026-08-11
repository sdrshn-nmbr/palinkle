import jax
import jax.numpy as jnp
import jax.lax as lax
from jax.experimental import pallas as pl
import jax.experimental.pallas.tpu as pltpu

def workload(x, weight, bias):
    def kernel(x_ref, w_ref, b_ref, out_ref):
        # Load tiles
        x_tile = x_ref[...].astype(jnp.float32)  # (128, 8192)
        w_tile = w_ref[...].astype(jnp.float32)  # (8192, 128)
        b_tile = b_ref[...].astype(jnp.float32)  # (128,)
        
        # Matmul + bias
        y = jnp.dot(x_tile, w_tile)  # (128, 128)
        y = y + b_tile[None, :]  # broadcast bias
        
        # Scaling
        y = y * 0.5
        
        # Hardtanh
        y = jnp.clip(y, -2.0, 2.0)
        
        # GELU approximation
        # 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        # But the AST says: x * 0.5 * (1.0 + tanh(...))
        # Actually looking at AST: left = x * 0.5, right = (1.0 + tanh(...))
        # So y = (y * 0.5) * (1.0 + tanh(...))
        # Wait, let me re-read the AST.
        
        # The last assignment is:
        # left = BinOp(left=BinOp(left=x, op=Mult, right=0.5), op=Mult, right=BinOp(...))
        # So it's (x * 0.5) * (1.0 + tanh(...))
        
        # Actually wait, the previous step was clip, so x is clipped.
        # Then: x = (x * 0.5) * (1.0 + tanh(sqrt(2/pi) * (x + 0.044715 * x**3)))
        
        # Let me compute exactly as AST says:
        inner = y + 0.044715 * (y ** 3)
        tanh_arg = jnp.sqrt(2.0 / jnp.pi) * inner
        gelu_factor = 1.0 + jnp.tanh(tanh_arg)
        y = (y * 0.5) * gelu_factor
        
        out_ref[...] = y.astype(jnp.bfloat16)
    
    batch = 4096
    out_features = 8192
    block_m = 128
    block_n = 128
    
    grid = (batch // block_m, out_features // block_n)
    
    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        grid=grid,
        in_specs=(
            pl.BlockSpec((block_m, 8192), lambda i, j: (i * block_m, 0)),
            pl.BlockSpec((8192, block_n), lambda i, j: (0, j * block_n)),
            pl.BlockSpec((block_n,), lambda i, j: (j * block_n,)),
        ),
        out_specs=pl.BlockSpec((block_m, block_n), lambda i, j: (i * block_m, j * block_n)),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel", "parallel")),
    )(x, weight, bias)
