# 48k_Matmul_BatchNorm_BiasAdd_Divide_Swish

Implement this pinned JAXBench workload as a normally lowered TPU Pallas kernel. Preserve the original workload semantics and original deployment shapes. The hidden verifier creates the exact inputs, computes the JAX baseline, checks normal Pallas lowering, captures a TPU profile, and compares performance against XLA.

97_Matmul_BatchNorm_BiasAdd_Divide_Swish — JAXBench fused operator workload.

## Configuration

```json
{
  "batch_size": 4096,
  "bn_eps": 1e-05,
  "bn_momentum": 0.1,
  "divide_value": 1.0,
  "in_features": 8192,
  "name": "97_Matmul_BatchNorm_BiasAdd_Divide_Swish",
  "out_features": 8192
}
```

## Required interface

```python
def workload(x, weight, linear_bias, bn_scale, bn_bias, bn_mean, bn_var, bias):
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
      "name": "linear_bias",
      "shape": [
        8192
      ]
    },
    {
      "dtype": "bfloat16",
      "name": "bn_scale",
      "shape": [
        8192
      ]
    },
    {
      "dtype": "bfloat16",
      "name": "bn_bias",
      "shape": [
        8192
      ]
    },
    {
      "dtype": "bfloat16",
      "name": "bn_mean",
      "shape": [
        8192
      ]
    },
    {
      "dtype": "bfloat16",
      "name": "bn_var",
      "shape": [
        8192
      ]
    },
    {
      "dtype": "bfloat16",
      "name": "bias",
      "shape": [
        1
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
    "linear_bias",
    "bn_scale",
    "bn_bias",
    "bn_mean",
    "bn_var",
    "bias"
  ],
  "body": [
    {
      "kind": "Assign",
      "targets": [
        {
          "id": "bn_eps",
          "kind": "Name"
        }
      ],
      "value": {
        "kind": null,
        "value": 1e-05
      }
    },
    {
      "kind": "Assign",
      "targets": [
        {
          "id": "divide_value",
          "kind": "Name"
        }
      ],
      "value": {
        "kind": null,
        "value": 1.0
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
          "id": "linear_bias",
          "kind": "Name"
        }
      }
    },
    {
      "kind": "Assign",
      "targets": [
        {
          "id": "x_normalized",
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
            "kind": "Sub"
          },
          "right": {
            "id": "bn_mean",
            "kind": "Name"
          }
        },
        "op": {
          "kind": "Div"
        },
        "right": {
          "args": [
            {
              "kind": "BinOp",
              "left": {
                "id": "bn_var",
                "kind": "Name"
              },
              "op": {
                "kind": "Add"
              },
              "right": {
                "id": "bn_eps",
                "kind": "Name"
              }
            }
          ],
          "func": {
            "attr": "sqrt",
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
          "kind": "BinOp",
          "left": {
            "id": "bn_scale",
            "kind": "Name"
          },
          "op": {
            "kind": "Mult"
          },
          "right": {
            "id": "x_normalized",
            "kind": "Name"
          }
        },
        "op": {
          "kind": "Add"
        },
        "right": {
          "id": "bn_bias",
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
          "id": "divide_value",
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
              "id": "x",
              "kind": "Name"
            }
          ],
          "func": {
            "attr": "sigmoid",
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

Matmul + BatchNorm + BiasAdd + Divide + Swish.

The input generator, concrete test values, baseline implementation, optimized reference, correctness tests, and grading logic are hidden. Do not use `interpret=True` or a plain-JAX fallback.
