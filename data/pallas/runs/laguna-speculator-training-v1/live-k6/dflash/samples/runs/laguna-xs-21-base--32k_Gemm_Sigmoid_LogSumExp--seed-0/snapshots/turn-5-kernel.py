import jax
import jax.numpy as jnp
import jax.pallas as pl
import jax.pallas as plp
import jax.interpreters.tpu as pltpu


def kernel(x_ref, w1_ref, b1_ref, w2_ref, b2_ref, out_ref, *, M, K1, N1, K2, N2):
    """Pallas kernel for Gemm + Sigmoid + Gemm + LogSumExp."""
    # Grid dimensions: M = batch_size = 16384
    # We process the entire matrix in one kernel
    
    # Initialize accumulator for first matmul in float32
    acc1 = jnp.zeros((K1,), dtype=jnp.float32)
    
    # First matmul: x @ w1.T + b1
    # x_ref has shape (M, K1) where K1 = 2048
    # w1_ref has shape (N1, K1) where N1 = 4096
    # We need to compute (M, N1) output
    
    # Accumulate along K dimension
    for k in range(K1):
        acc1 = acc1 + x_ref[k].astype(jnp.float32) * w1_ref[:, k].astype(jnp.float32)
    
    # Add bias and apply sigmoid
    acc1 = acc1 + b1_ref.astype(jnp.float32)
    acc1 = jax.nn.sigmoid(acc1)
    
    # Second matmul: acc1 @ w2.T + b2
    # acc1 has shape (N1,) = (4096,)
    # w2_ref has shape (N2, N1) where N2 = 1024
    # We need to compute (N2,) output
    
    acc2 = jnp.zeros((N2,), dtype=jnp.float32)
    for n1 in range(N1):
        acc2 = acc2 + acc1[n1] * w2_ref[:, n1].astype(jnp.float32)
    
    # Add bias
    acc2 = acc2 + b2_ref.astype(jnp.float32)
    
    # LogSumExp along axis 0 (since we have a 1D result)
    # For a 1D array, logsumexp is just the log-sum-exp of all elements
    result = jnp.log(jnp.sum(jnp.exp(acc2)))
    
    out_ref[...] = result.astype(jnp.bfloat16)


def workload(x, w1, b1, w2, b2):
    """Workload: Gemm + Sigmoid + Gemm + LogSumExp."""
    M = x.shape[0]  # batch_size = 16384
    K1 = x.shape[1]  # input_size = 2048
    N1 = w1.shape[0]  # hidden_size = 4096
    K2 = w2.shape[1]  # hidden_size = 4096
    N2 = w2.shape[0]  # output_size = 1024
    
    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((N2,), x.dtype),
        grid=(1,),
        in_specs=(
            pl.BlockSpec((K1,), lambda i: (0,)),  # x: first row
            pl.BlockSpec((N1, K1), lambda i: (0, slice(None))),  # w1
            pl.BlockSpec((N1,), lambda i: (0,)),  # b1
            pl.BlockSpec((N2, N1), lambda i: (slice(None), 0)),  # w2
            pl.BlockSpec((N2,), lambda i: (0,)),  # b2
        ),
        out_specs=pl.BlockSpec((N2,), lambda i: (0,)),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel",)
        ),
        interpret=False,
        M=M, K1=K1, N1=N1, K2=K2, N2=N2,
    )(x, w1, b1, w2, b2)
