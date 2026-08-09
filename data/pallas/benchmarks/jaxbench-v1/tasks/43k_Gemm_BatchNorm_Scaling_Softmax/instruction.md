# 43k_Gemm_BatchNorm_Scaling_Softmax

Implement this pinned JAXBench workload as a normally lowered TPU Pallas kernel. Preserve the original workload semantics and original deployment shapes. The hidden verifier creates the exact inputs, computes the JAX baseline, checks normal Pallas lowering, captures a TPU profile, and compares performance against XLA.

84_Gemm_BatchNorm_Scaling_Softmax — JAXBench fused operator workload.

## Configuration

```json
{
  "batch_size": 4096,
  "bn_eps": 1e-05,
  "bn_momentum": 0.1,
  "in_features": 8192,
  "name": "84_Gemm_BatchNorm_Scaling_Softmax",
  "out_features": 8192
}
```

## Required interface

```python
def workload(x, weight, bias, bn_scale, bn_bias, bn_mean, bn_var, scale):
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
    },
    {
      "dtype": "float32",
      "name": "bn_scale",
      "shape": [
        8192
      ]
    },
    {
      "dtype": "float32",
      "name": "bn_bias",
      "shape": [
        8192
      ]
    },
    {
      "dtype": "float32",
      "name": "bn_mean",
      "shape": [
        8192
      ]
    },
    {
      "dtype": "float32",
      "name": "bn_var",
      "shape": [
        8192
      ]
    },
    {
      "dtype": "float32",
      "name": "scale",
      "shape": [
        1
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
    "bias",
    "bn_scale",
    "bn_bias",
    "bn_mean",
    "bn_var",
    "scale"
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
          "id": "scale",
          "kind": "Name"
        },
        "op": {
          "kind": "Mult"
        },
        "right": {
          "id": "x",
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
        "args": [
          {
            "id": "x",
            "kind": "Name"
          }
        ],
        "func": {
          "attr": "softmax",
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
        "keywords": [
          {
            "arg": "axis",
            "kind": "keyword",
            "value": {
              "kind": null,
              "value": 1
            }
          }
        ],
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
    "jax": "jax",
    "jnp": "jax.numpy"
  },
  "module_values": {},
  "unresolved_names": []
}
```

Gemm + BatchNorm + Scaling + Softmax.

The input generator, concrete test values, baseline implementation, optimized reference, correctness tests, and grading logic are hidden. Do not use `interpret=True` or a plain-JAX fallback.
