# Pinned JAX Pallas TPU API

The development environment contains JAX and JAXlib 0.10.1. The hidden TPU
runtime uses the same versions. Implement `workload(*inputs)` with normal Pallas
lowering. Do not use `interpret=True` or a plain-JAX result path.

## Core interfaces

```python
pl.pallas_call(
    kernel,
    out_shape,
    *,
    grid=(),
    grid_spec=None,
    in_specs=pl.no_block_spec,
    out_specs=pl.no_block_spec,
    scratch_shapes=(),
    compiler_params=None,
    interpret=False,
)(*inputs)

pl.BlockSpec(block_shape, index_map, *, memory_space=None)
pl.program_id(axis)
pltpu.PrefetchScalarGridSpec(
    num_scalar_prefetch,
    in_specs,
    out_specs,
    grid,
    scratch_shapes=(),
)
pltpu.CompilerParams(dimension_semantics=(...))
pltpu.VMEM(shape, dtype)
```

Kernel arguments are references. Read and write them with indexing:

```python
def add_kernel(x_ref, y_ref, out_ref):
    out_ref[...] = x_ref[...] + y_ref[...]

def workload(x, y):
    block = 128
    return pl.pallas_call(
        add_kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        grid=(x.size // block,),
        in_specs=(
            pl.BlockSpec((block,), lambda i: (i,)),
            pl.BlockSpec((block,), lambda i: (i,)),
        ),
        out_specs=pl.BlockSpec((block,), lambda i: (i,)),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel",)
        ),
    )(x, y)
```

For TPU matmul, use `jnp.dot` inside the Pallas kernel and accumulate in
float32 VMEM. TPU block dimensions generally need multiples of 8 for bf16 and
128-element tiling along vectorized dimensions. Use `pl.when` for conditional
initialization and prefer static grids and block shapes.

Run `python dev_check.py` after each edit. It parses and imports `kernel.py`,
traces `workload` against the public tensor schema, and requires a reachable
Pallas primitive. This does not replace hidden TPU compilation or correctness.
