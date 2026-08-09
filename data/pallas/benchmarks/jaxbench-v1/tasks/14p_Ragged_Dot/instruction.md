# 14p_Ragged_Dot

Implement this pinned JAXBench workload as a normally lowered TPU Pallas kernel. Preserve the original workload semantics and original deployment shapes. The hidden verifier creates the exact inputs, computes the JAX baseline, checks normal Pallas lowering, captures a TPU profile, and compares performance against XLA.

Grouped Matmul (Ragged Dot) for MoE — Mixtral 8x7B. From openxla/tokamax.

## Configuration

```json
{
  "K": 4096,
  "M": 8192,
  "N": 14336,
  "model": "Mixtral-8x7B",
  "name": "mixtral_8x7b_ragged_dot",
  "num_groups": 8,
  "operator": "ragged_dot"
}
```

## Required interface

```python
def workload(x, weights):
    ...
```

## Tensor contract

```json
{
  "inputs": [
    {
      "dtype": "bfloat16",
      "name": "x",
      "shape": [
        8,
        1024,
        4096
      ]
    },
    {
      "dtype": "bfloat16",
      "name": "weights",
      "shape": [
        8,
        4096,
        14336
      ]
    }
  ],
  "outputs": [
    {
      "dtype": "bfloat16",
      "shape": [
        8,
        1024,
        14336
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
    "x",
    "weights"
  ],
  "body": [
    {
      "kind": "Return",
      "value": {
        "args": [
          {
            "kind": null,
            "value": "gmk,gkn->gmn"
          },
          {
            "id": "x",
            "kind": "Name"
          },
          {
            "id": "weights",
            "kind": "Name"
          }
        ],
        "func": {
          "attr": "einsum",
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

Grouped matmul: each group does independent matmul. Equivalent to ragged dot.

The input generator, concrete test values, baseline implementation, optimized reference, correctness tests, and grading logic are hidden. Do not use `interpret=True` or a plain-JAX fallback.
