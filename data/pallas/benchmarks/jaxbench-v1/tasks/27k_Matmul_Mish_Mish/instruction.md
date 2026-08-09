# 27k_Matmul_Mish_Mish

Implement this pinned JAXBench workload as a normally lowered TPU Pallas kernel. Preserve the original workload semantics and original deployment shapes. The hidden verifier creates the exact inputs, computes the JAX baseline, checks normal Pallas lowering, captures a TPU profile, and compares performance against XLA.

29_Matmul_Mish_Mish — JAXBench fused operator workload.

## Configuration

```json
{
  "batch_size": 4096,
  "in_features": 8192,
  "name": "29_Matmul_Mish_Mish",
  "out_features": 8192
}
```

## Required interface

```python
def workload(x, weight, bias):
    ...
```

## Tensor contract

```json
{
  "inputs": [
    {
      "dtype": "float32",
      "name": "x",
      "shape": [
        4096,
        8192
      ]
    },
    {
      "dtype": "float32",
      "name": "weight",
      "shape": [
        8192,
        8192
      ]
    },
    {
      "dtype": "float32",
      "name": "bias",
      "shape": [
        8192
      ]
    }
  ],
  "outputs": [
    {
      "dtype": "float32",
      "shape": [
        4096,
        8192
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
    "weight",
    "bias"
  ],
  "body": [
    {
      "kind": "Assign",
      "targets": [
        {
          "id": "x",
          "kind": "Name"
        }
      ],
      "value": {
        "kind": "BinOp",
        "left": {
          "kind": "BinOp",
          "left": {
            "id": "x",
            "kind": "Name"
          },
          "op": {
            "kind": "MatMult"
          },
          "right": {
            "id": "weight",
            "kind": "Name"
          }
        },
        "op": {
          "kind": "Add"
        },
        "right": {
          "id": "bias",
          "kind": "Name"
        }
      }
    },
    {
      "kind": "Assign",
      "targets": [
        {
          "id": "x",
          "kind": "Name"
        }
      ],
      "value": {
        "kind": "BinOp",
        "left": {
          "id": "x",
          "kind": "Name"
        },
        "op": {
          "kind": "Mult"
        },
        "right": {
          "args": [
            {
              "args": [
                {
                  "id": "x",
                  "kind": "Name"
                }
              ],
              "func": {
                "attr": "softplus",
                "kind": "Attribute",
                "value": {
                  "attr": "nn",
                  "kind": "Attribute",
                  "value": {
                    "id": "jax",
                    "kind": "Name"
                  }
                }
              },
              "keywords": [],
              "kind": "Call"
            }
          ],
          "func": {
            "attr": "tanh",
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
    },
    {
      "kind": "Assign",
      "targets": [
        {
          "id": "x",
          "kind": "Name"
        }
      ],
      "value": {
        "kind": "BinOp",
        "left": {
          "id": "x",
          "kind": "Name"
        },
        "op": {
          "kind": "Mult"
        },
        "right": {
          "args": [
            {
              "args": [
                {
                  "id": "x",
                  "kind": "Name"
                }
              ],
              "func": {
                "attr": "softplus",
                "kind": "Attribute",
                "value": {
                  "attr": "nn",
                  "kind": "Attribute",
                  "value": {
                    "id": "jax",
                    "kind": "Name"
                  }
                }
              },
              "keywords": [],
              "kind": "Call"
            }
          ],
          "func": {
            "attr": "tanh",
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
    },
    {
      "kind": "Return",
      "value": {
        "id": "x",
        "kind": "Name"
      }
    }
  ],
  "format": "canonical_python_ast_semantics_v1",
  "helper_functions": {},
  "imports": {
    "jax": "jax",
    "jnp": "jax.numpy"
  },
  "module_values": {},
  "unresolved_names": []
}
```

Matmul + Mish + Mish.

The input generator, concrete test values, baseline implementation, optimized reference, correctness tests, and grading logic are hidden. Do not use `interpret=True` or a plain-JAX fallback.
