# 20k_Gemm_Multiply_LeakyReLU

Implement this pinned JAXBench workload as a normally lowered TPU Pallas kernel. Preserve the original workload semantics and original deployment shapes. The hidden verifier creates the exact inputs, computes the JAX baseline, checks normal Pallas lowering, captures a TPU profile, and compares performance against XLA.

12_Gemm_Multiply_LeakyReLU — JAXBench fused operator workload.

## Configuration

```json
{
  "batch_size": 4096,
  "in_features": 8192,
  "multiplier": 2.0,
  "name": "12_Gemm_Multiply_LeakyReLU",
  "negative_slope": 0.1,
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
      "dtype": "bfloat16",
      "name": "x",
      "shape": [
        4096,
        8192
      ]
    },
    {
      "dtype": "bfloat16",
      "name": "weight",
      "shape": [
        8192,
        8192
      ]
    },
    {
      "dtype": "bfloat16",
      "name": "bias",
      "shape": [
        8192
      ]
    }
  ],
  "outputs": [
    {
      "dtype": "bfloat16",
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
          "args": [
            {
              "id": "x",
              "kind": "Name"
            },
            {
              "id": "weight",
              "kind": "Name"
            }
          ],
          "func": {
            "attr": "matmul",
            "kind": "Attribute",
            "value": {
              "id": "jnp",
              "kind": "Name"
            }
          },
          "keywords": [],
          "kind": "Call"
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
          "kind": null,
          "value": 2.0
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
        "args": [
          {
            "comparators": [
              {
                "kind": null,
                "value": 0
              }
            ],
            "kind": "Compare",
            "left": {
              "id": "x",
              "kind": "Name"
            },
            "ops": [
              {
                "kind": "GtE"
              }
            ]
          },
          {
            "id": "x",
            "kind": "Name"
          },
          {
            "kind": "BinOp",
            "left": {
              "id": "x",
              "kind": "Name"
            },
            "op": {
              "kind": "Mult"
            },
            "right": {
              "kind": null,
              "value": 0.1
            }
          }
        ],
        "func": {
          "attr": "where",
          "kind": "Attribute",
          "value": {
            "id": "jnp",
            "kind": "Name"
          }
        },
        "keywords": [],
        "kind": "Call"
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
    "jnp": "jax.numpy"
  },
  "module_values": {},
  "unresolved_names": []
}
```

Gemm + Multiply + LeakyReLU.

The input generator, concrete test values, baseline implementation, optimized reference, correctness tests, and grading logic are hidden. Do not use `interpret=True` or a plain-JAX fallback.
