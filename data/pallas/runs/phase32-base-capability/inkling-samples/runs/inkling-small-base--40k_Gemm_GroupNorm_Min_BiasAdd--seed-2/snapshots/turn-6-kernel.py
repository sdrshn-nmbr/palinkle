import jax
import jax.numpy as jnp
import jax.experimental.pallas as pl
import jax.experimental.pallas.tpu as pltpu

def workload(x, weight, linear_bias, gn_weight, gn_bias, bias):
    def kernel(x_ref, w_ref, lb_ref, gn_w_ref, gn_b_ref, b_ref, out_ref):
        # Load full arrays from refs
        x_val = x_ref[...]
        w_val = w_ref[...]
        lb_val = lb_ref[...]
        gn_w_val = gn_w_ref[...]
        gn_b_val = gn_b_ref[...]
        b_val = b_ref[...]
        
        # Step 1: matmul + linear_bias
        x_val = jnp.dot(x_val, w_val.T) + lb_val
        
        # Step 2: get shape
        N, C = x_val.shape
        
        # Step 3: group norm setup
        num_groups = 512
        eps = 1e-05
        G = num_groups
        
        # Step 4: reshape for group norm
        x_val = jnp.reshape(x_val, (N, G, C // G))
        
        # Step 5: mean and var over axis=2
        mean = jnp.mean(x_val, axis=2, keepdims=True)
        var = jnp.var(x_val, axis=2, keepdims=True)
        
        # Step 6: normalize
        x_val = (x_val - mean) / jnp.sqrt(var + eps)
        
        # Step 7: reshape back
        x_val = jnp.reshape(x_val, (N, C))
        
        # Step 8: scale and shift
        x_val = x_val * gn_w_val + gn_b_val
        
        # Step 9: min over axis=1
        x_val = jnp.min(x_val, axis=1, keepdims=True)
        
        # Step 10: reshape to (1, 1, N, 1)
        x_val = jnp.reshape(x_val, (1, 1, N, 1))
        
        # Step 11: add bias
        x_val = x_val + b_val
        
        out_ref[...] = x_val
    
    out_shape = jax.ShapeDtypeStruct((1, 8192, 4096, 1), jnp.bfloat16)
    
    return pl.pallas_call(
        kernel,
        out_shape=out_shape,
        grid=(1,),
        in_specs=(
            pl.BlockSpec((4096, 8192), lambda i: (0, 0)),
            pl.BlockSpec((8192, 8192), lambda i: (0, 0)),
            pl.BlockSpec((8192,), lambda i: (0,)),
            pl.BlockSpec((8192,), lambda i: (0,)),
            pl.BlockSpec((8192,), lambda i: (0,)),
            pl.BlockSpec((1, 8192, 1, 1), lambda i: (0, 0, 0, 0)),
        ),
        out_specs=pl.BlockSpec((1, 8192, 4096, 1), lambda i: (0, 0, 0, 0)),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel",)),
    )(x, weight, linear_bias, gn_weight, gn_bias, bias)
