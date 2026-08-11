import jax
import jax.numpy as jnp
import pallas as pl
import pallas.lib as pllib
import jax.pallas as pl

def kernel(
    x_ref,
    weight_ref,
    linear_bias_ref,
    gn_weight_ref,
    gn_bias_ref,
    bias_ref,
    out_ref,
):
    # x_ref has shape [4096, 8192]
    # weight_ref has shape [8192, 8192]
    # linear_bias_ref has shape [8192]
    # gn_weight_ref has shape [8192]
    # gn_bias_ref has shape [8192]
    # bias_ref has shape [1, 8192, 1, 1]
    # out_ref has shape [1, 8192, 4096, 1]
    
    # Step 1: matmul(x, weight.T) + linear_bias
    x = x_ref[:]
    weight = weight_ref[:]
    linear_bias = linear_bias_ref[:]
    
    # Compute matmul result
    # x shape: [4096, 8192], weight.T shape: [8192, 8192]
    # result shape: [4096, 8192]
    x = jnp.dot(x, weight.T) + linear_bias
    
    # Step 2: reshape to [N, G, C//G]
    N, C = x.shape
    G = 512
    x = x.reshape(N, G, C // G)
    
    # Step 3: GroupNorm
    eps = 1e-5
    mean = jnp.mean(x, axis=2, keepdims=True)
    var = jnp.var(x, axis=2, keepdims=True)
    x = (x - mean) / jnp.sqrt(var + eps)
    
    # Step 4: Scale and shift with group norm parameters
    gn_weight = gn_weight_ref[:].reshape(1, -1, 1)
    gn_bias = gn_bias_ref[:].reshape(1, -1, 1)
    x = x * gn_weight + gn_bias
    
    # Step 5: Min along axis 1
    x = jnp.min(x, axis=1, keepdims=True)
    
    # Step 6: Reshape to [N, C]
    x = x.reshape(N, C)
    
    # Step 7: Reshape to [1, C, N, 1]
    x = x.reshape(1, C, N, 1)
    
    # Step 8: Add bias
    bias = bias_ref[:]
    x = x + bias
    
    # Write output
    out_ref[:] = x


def workload(x, weight, linear_bias, gn_weight, gn_bias, bias):
    """Fused GEMM + GroupNorm + Min + BiasAdd kernel."""
    
    # Define block shapes for Pallas
    # We'll process the input x in blocks
    block_size = 128  # Process 128 elements at a time
    
    # Grid spec for the kernel
    grid = (x.shape[0], x.shape[1] // block_size)
    
    # Call the pallas kernel
    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((1, 8192, 4096, 1), jax.dtypes.bfloat16),
        grid=grid,
        in_specs=(
            pl.BlockSpec((x.shape[0], x.shape[1]), lambda _: (0, 0)),
            pl.BlockSpec((weight.shape[0], weight.shape[1]), lambda _: (0, 0)),
            pl.BlockSpec((linear_bias.shape[0],), lambda _: (0,)),
            pl.BlockSpec((gn_weight.shape[0],), lambda _: (0,)),
            pl.BlockSpec((gn_bias.shape[0],), lambda _: (0,)),
            pl.BlockSpec((bias.shape[0], bias.shape[1], bias.shape[2], bias.shape[3]), lambda _: (0, 0, 0, 0)),
        ),
        out_specs=pl.BlockSpec((1, 8192, 4096, 1), lambda _: (0, 0, 0, 0)),
        compiler_params=pllib.CompilerParams(
            dimension_semantics=("parallel", "parallel")
        ),
    )(x, weight, linear_bias, gn_weight, gn_bias, bias)
