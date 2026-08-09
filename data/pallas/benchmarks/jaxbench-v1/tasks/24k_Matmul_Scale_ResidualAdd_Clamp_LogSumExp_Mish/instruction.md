# 24k_Matmul_Scale_ResidualAdd_Clamp_LogSumExp_Mish

Implement this pinned JAXBench workload as a normally lowered TPU Pallas kernel. Preserve the original workload semantics and original deployment shapes. The hidden verifier creates the exact inputs, computes the JAX baseline, checks normal Pallas lowering, captures a TPU profile, and compares performance against XLA.

22_Matmul_Scale_ResidualAdd_Clamp_LogSumExp_Mish — JAXBench fused operator workload.

## Configuration

```json
{
  "batch_size": 4096,
  "clamp_max": 10.0,
  "clamp_min": -10.0,
  "hidden_size": 8192,
  "input_size": 8192,
  "name": "22_Matmul_Scale_ResidualAdd_Clamp_LogSumExp_Mish",
  "scale_factor": 2.0
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
        1
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
              "attr": "T",
              "kind": "Attribute",
              "value": {
                "id": "weight",
                "kind": "Name"
              }
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
        "kind": "BinOp",
        "left": {
          "id": "x",
          "kind": "Name"
        },
        "op": {
          "kind": "Add"
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
          },
          {
            "kind": "UnaryOp",
            "op": {
              "kind": "USub"
            },
            "operand": {
              "kind": null,
              "value": 10.0
            }
          },
          {
            "kind": null,
            "value": 10.0
          }
        ],
        "func": {
          "attr": "clip",
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
          "attr": "logsumexp",
          "kind": "Attribute",
          "value": {
            "attr": "special",
            "kind": "Attribute",
            "value": {
              "attr": "scipy",
              "kind": "Attribute",
              "value": {
                "id": "jax",
                "kind": "Name"
              }
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
          },
          {
            "arg": "keepdims",
            "kind": "keyword",
            "value": {
              "kind": null,
              "value": true
            }
          }
        ],
        "kind": "Call"
      }
    },
    {
      "kind": "Assign",
      "targets": [
        {
          "id": "softplus_x",
          "kind": "Name"
        }
      ],
      "value": {
        "args": [
          {
            "id": "x",
            "kind": "Name"
          },
          {
            "kind": null,
            "value": 0.0
          }
        ],
        "func": {
          "attr": "logaddexp",
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
      "kind": "Assign",
      "targets": [
        {
          "id": "mish_x",
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
              "id": "softplus_x",
              "kind": "Name"
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
          "id": "mish_x",
          "kind": "Name"
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

Matmul + Scale + ResidualAdd + Clamp + LogSumExp + Mish.

The input generator, concrete test values, baseline implementation, optimized reference, correctness tests, and grading logic are hidden. Do not use `interpret=True` or a plain-JAX fallback.
