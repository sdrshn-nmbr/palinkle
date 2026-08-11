import jax
import jax.numpy as jnp
import jax.experimental.pallas as pl
import jax.experimental.pallas.tpu as pltpu

def workload(x, weight, bias, add_value):
    batch_size, in_features = x.shape
    out_features = weight.shape[1]
    block_batch = 128
    block_k = 128
    
    def kernel(x_ref, w_ref, b_ref, a_ref, out_ref):
        # Initialize float32 accumulator
        acc = jnp.zeros((block_batch, out_features), dtype=jnp.float32)
        
        # Loop over contraction dimension
        for k in range(0, in_features, block_k):
            x_slice = x_ref[:, k:k+block_k]
            w_slice = w_ref[k:k+block_k, :]
            acc += jnp.dot(x_slice, w_slice)
        
        # Add bias and add_value (broadcast over batch)
        acc += b_ref[None, :]
        acc += a_ref[None, :]
        
        # Apply swish: x * sigmoid(x)
        swish = acc * jax.nn.sigmoid(acc)
        # tanh
        tanh = jnp.tanh(swish)
        # gelu
        gelu = jax.nn.gelu(tanh)
        # hardtanh: clip to [-1, 1]
        result = jnp.clip(gelu, -1.0, 1.0)
        
        out_ref[...] = result.astype(jnp.bfloat16)
    
    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, jnp.bfloat16),
        grid=(batch_size // block_batch,),
        in_specs=(
            pl.BlockSpec((block_batch, in_features), lambda i: (i * block_batch, 0)),
            pl.BlockSpec((in_features, out_features), lambda i: (0, 0)),
            pl.BlockSpec((out_features,), lambda i: (0,)),
            pl.BlockSpec((out_features,), lambda i: (0,)),
        ),
        out_specs=pl.BlockSpec((block_batch, out_features), lambda i: (i * block_batch, 0)),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel",)),
    )(x, weight, bias, add_value)
