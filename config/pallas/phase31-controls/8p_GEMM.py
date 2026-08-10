import jax
from jax.experimental.pallas.ops.tpu.matmul import matmul


def workload(x, y):
    return matmul(
        x,
        y,
        block_shape=(1024, 2048),
        block_k=1024,
        out_dtype=jax.numpy.bfloat16,
    )
