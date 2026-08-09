# 8p_GEMM

Implement this pinned JAXBench workload as a normally lowered TPU Pallas kernel. Preserve the original workload semantics and original deployment shapes. The hidden verifier creates the exact inputs, computes the JAX baseline, checks normal Pallas lowering, captures a TPU profile, and compares performance against XLA.

Dense bf16 GEMM — Llama-70B FFN dimensions.

Baseline dense matrix multiplication at Llama-3.1-70B hidden-to-FFN scale.
A = (8192, 8192), B = (8192, 28672) — matches hidden_dim -> mlp_dim projection.

## Configuration

```json
{
  "K": 8192,
  "M": 8192,
  "N": 28672,
  "model": "Llama-3.1-70B",
  "name": "gemm_llama70b",
  "operator": "dense_matmul"
}
```

## Required interface

```python
def workload(A, B):
    ...
```

## Tensor contract

```json
{
  "inputs": [
    {
      "dtype": "bfloat16",
      "name": "A",
      "shape": [
        8192,
        8192
      ]
    },
    {
      "dtype": "bfloat16",
      "name": "B",
      "shape": [
        8192,
        28672
      ]
    }
  ],
  "outputs": [
    {
      "dtype": "bfloat16",
      "shape": [
        8192,
        28672
      ]
    }
  ]
}
```

## Exact semantic contract

The following non-executable canonical AST defines operation order, constants, axes, layouts, padding, precision, and all other observable semantics. `Name` and `Attribute` nodes name mathematical/JAX operations; the hidden source implementation is not included.

```json
{
  "arguments": [
    "A",
    "B"
  ],
  "body": [
    {
      "kind": "Return",
      "value": {
        "args": [
          {
            "id": "A",
            "kind": "Name"
          },
          {
            "id": "B",
            "kind": "Name"
          }
        ],
        "func": {
          "attr": "dot",
          "kind": "Attribute",
          "value": {
            "id": "jnp",
            "kind": "Name"
          }
        },
        "keywords": [],
        "kind": "Call"
      }
    }
  ],
  "format": "canonical_python_ast_semantics_v1",
  "helper_functions": {},
  "imports": {
    "jnp": "jax.numpy"
  },
  "module_values": {},
  "unresolved_names": []
}
```

Dense matmul: C = A @ B

The input generator, concrete test values, baseline implementation, optimized reference, correctness tests, and grading logic are hidden. Do not use `interpret=True` or a plain-JAX fallback.
