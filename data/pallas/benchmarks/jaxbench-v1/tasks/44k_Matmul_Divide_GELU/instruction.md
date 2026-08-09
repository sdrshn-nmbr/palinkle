# 44k_Matmul_Divide_GELU

Implement this pinned JAXBench workload as a normally lowered TPU Pallas kernel. Preserve the original workload semantics and original deployment shapes. The hidden verifier creates the exact inputs, computes the JAX baseline, checks normal Pallas lowering, captures a TPU profile, and compares performance against XLA.

86_Matmul_Divide_GELU — JAXBench fused operator workload.

## Configuration

```json
{
  "batch_size": 4096,
  "divisor": 10.0,
  "input_size": 8192,
  "name": "86_Matmul_Divide_GELU",
  "output_size": 8192
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
          "kind": "Div"
        },
        "right": {
          "kind": null,
          "value": 10.0
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
            "id": "x",
            "kind": "Name"
          }
        ],
        "func": {
          "attr": "gelu",
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
    "jax": "jax"
  },
  "module_values": {},
  "unresolved_names": []
}
```

Matmul + Divide + GELU.

The input generator, concrete test values, baseline implementation, optimized reference, correctness tests, and grading logic are hidden. Do not use `interpret=True` or a plain-JAX fallback.
