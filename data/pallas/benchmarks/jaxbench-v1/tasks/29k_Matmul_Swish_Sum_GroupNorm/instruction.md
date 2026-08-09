# 29k_Matmul_Swish_Sum_GroupNorm

Implement this pinned JAXBench workload as a normally lowered TPU Pallas kernel. Preserve the original workload semantics and original deployment shapes. The hidden verifier creates the exact inputs, computes the JAX baseline, checks normal Pallas lowering, captures a TPU profile, and compares performance against XLA.

37_Matmul_Swish_Sum_GroupNorm — JAXBench fused operator workload.

## Configuration

```json
{
  "batch_size": 8192,
  "in_features": 4096,
  "name": "37_Matmul_Swish_Sum_GroupNorm",
  "num_groups": 64,
  "out_features": 4096
}
```

## Required interface

```python
def workload(x, weight, bias, gn_weight, gn_bias):
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
        8192,
        4096
      ]
    },
    {
      "dtype": "float32",
      "name": "weight",
      "shape": [
        4096,
        4096
      ]
    },
    {
      "dtype": "float32",
      "name": "bias",
      "shape": [
        4096
      ]
    },
    {
      "dtype": "float32",
      "name": "gn_weight",
      "shape": [
        4096
      ]
    },
    {
      "dtype": "float32",
      "name": "gn_bias",
      "shape": [
        4096
      ]
    }
  ],
  "outputs": [
    {
      "dtype": "float32",
      "shape": [
        8192,
        4096
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
    "gn_weight",
    "gn_bias"
  ],
  "body": [
    {
      "kind": "Assign",
      "targets": [
        {
          "id": "num_groups",
          "kind": "Name"
        }
      ],
      "value": {
        "kind": null,
        "value": 64
      }
    },
    {
      "kind": "Assign",
      "targets": [
        {
          "id": "out_features",
          "kind": "Name"
        }
      ],
      "value": {
        "kind": null,
        "value": 4096
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
          "id": "group_size",
          "kind": "Name"
        }
      ],
      "value": {
        "kind": "BinOp",
        "left": {
          "id": "out_features",
          "kind": "Name"
        },
        "op": {
          "kind": "FloorDiv"
        },
        "right": {
          "id": "num_groups",
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
            "kind": "UnaryOp",
            "op": {
              "kind": "USub"
            },
            "operand": {
              "kind": null,
              "value": 1
            }
          },
          {
            "id": "num_groups",
            "kind": "Name"
          },
          {
            "id": "group_size",
            "kind": "Name"
          }
        ],
        "func": {
          "attr": "reshape",
          "kind": "Attribute",
          "value": {
            "id": "x",
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
          "id": "mean",
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
          "attr": "mean",
          "kind": "Attribute",
          "value": {
            "id": "jnp",
            "kind": "Name"
          }
        },
        "keywords": [
          {
            "arg": "axis",
            "kind": "keyword",
            "value": {
              "kind": "UnaryOp",
              "op": {
                "kind": "USub"
              },
              "operand": {
                "kind": null,
                "value": 1
              }
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
          "id": "var",
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
          "attr": "var",
          "kind": "Attribute",
          "value": {
            "id": "jnp",
            "kind": "Name"
          }
        },
        "keywords": [
          {
            "arg": "axis",
            "kind": "keyword",
            "value": {
              "kind": "UnaryOp",
              "op": {
                "kind": "USub"
              },
              "operand": {
                "kind": null,
                "value": 1
              }
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
            "kind": "Sub"
          },
          "right": {
            "id": "mean",
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
                "id": "var",
                "kind": "Name"
              },
              "op": {
                "kind": "Add"
              },
              "right": {
                "kind": null,
                "value": 1e-05
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
        "args": [
          {
            "kind": "UnaryOp",
            "op": {
              "kind": "USub"
            },
            "operand": {
              "kind": null,
              "value": 1
            }
          },
          {
            "id": "out_features",
            "kind": "Name"
          }
        ],
        "func": {
          "attr": "reshape",
          "kind": "Attribute",
          "value": {
            "id": "x",
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
        "kind": "BinOp",
        "left": {
          "kind": "BinOp",
          "left": {
            "id": "x",
            "kind": "Name"
          },
          "op": {
            "kind": "Mult"
          },
          "right": {
            "id": "gn_weight",
            "kind": "Name"
          }
        },
        "op": {
          "kind": "Add"
        },
        "right": {
          "id": "gn_bias",
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

Matmul + Swish + Sum(bias) + GroupNorm.

The input generator, concrete test values, baseline implementation, optimized reference, correctness tests, and grading logic are hidden. Do not use `interpret=True` or a plain-JAX fallback.
