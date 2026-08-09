# 45k_Gemm_GroupNorm_Swish_Multiply_Swish

Implement this pinned JAXBench workload as a normally lowered TPU Pallas kernel. Preserve the original workload semantics and original deployment shapes. The hidden verifier creates the exact inputs, computes the JAX baseline, checks normal Pallas lowering, captures a TPU profile, and compares performance against XLA.

88_Gemm_GroupNorm_Swish_Multiply_Swish — JAXBench fused operator workload.

## Configuration

```json
{
  "batch_size": 4096,
  "in_features": 8192,
  "name": "88_Gemm_GroupNorm_Swish_Multiply_Swish",
  "num_groups": 256,
  "out_features": 8192
}
```

## Required interface

```python
def workload(x, gemm_weight, gemm_bias, gn_weight, gn_bias, multiply_weight):
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
      "name": "gemm_weight",
      "shape": [
        8192,
        8192
      ]
    },
    {
      "dtype": "bfloat16",
      "name": "gemm_bias",
      "shape": [
        8192
      ]
    },
    {
      "dtype": "bfloat16",
      "name": "gn_weight",
      "shape": [
        8192
      ]
    },
    {
      "dtype": "bfloat16",
      "name": "gn_bias",
      "shape": [
        8192
      ]
    },
    {
      "dtype": "bfloat16",
      "name": "multiply_weight",
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
    "gemm_weight",
    "gemm_bias",
    "gn_weight",
    "gn_bias",
    "multiply_weight"
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
        "value": 256
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
        "value": 8192
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
              "attr": "T",
              "kind": "Attribute",
              "value": {
                "id": "gemm_weight",
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
          "id": "gemm_bias",
          "kind": "Name"
        }
      }
    },
    {
      "kind": "Assign",
      "targets": [
        {
          "id": "batch_size",
          "kind": "Name"
        }
      ],
      "value": {
        "kind": "Subscript",
        "slice": {
          "kind": null,
          "value": 0
        },
        "value": {
          "attr": "shape",
          "kind": "Attribute",
          "value": {
            "id": "x",
            "kind": "Name"
          }
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
          "id": "x_grouped",
          "kind": "Name"
        }
      ],
      "value": {
        "args": [
          {
            "id": "batch_size",
            "kind": "Name"
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
            "id": "x_grouped",
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
            "id": "x_grouped",
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
          "id": "x_normalized",
          "kind": "Name"
        }
      ],
      "value": {
        "kind": "BinOp",
        "left": {
          "kind": "BinOp",
          "left": {
            "id": "x_grouped",
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
            "id": "batch_size",
            "kind": "Name"
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
            "id": "x_normalized",
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
          "id": "multiply_weight",
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

Gemm + GroupNorm + Swish + Multiply + Swish.

The input generator, concrete test values, baseline implementation, optimized reference, correctness tests, and grading logic are hidden. Do not use `interpret=True` or a plain-JAX fallback.
