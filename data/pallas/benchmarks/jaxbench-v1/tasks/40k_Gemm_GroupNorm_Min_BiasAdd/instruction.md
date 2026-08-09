# 40k_Gemm_GroupNorm_Min_BiasAdd

Implement this pinned JAXBench workload as a normally lowered TPU Pallas kernel. Preserve the original workload semantics and original deployment shapes. The hidden verifier creates the exact inputs, computes the JAX baseline, checks normal Pallas lowering, captures a TPU profile, and compares performance against XLA.

75_Gemm_GroupNorm_Min_BiasAdd — JAXBench fused operator workload.

## Configuration

```json
{
  "batch_size": 4096,
  "in_features": 8192,
  "name": "75_Gemm_GroupNorm_Min_BiasAdd",
  "num_groups": 512,
  "out_features": 8192
}
```

## Required interface

```python
def workload(x, weight, linear_bias, gn_weight, gn_bias, bias):
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
      "name": "linear_bias",
      "shape": [
        8192
      ]
    },
    {
      "dtype": "float32",
      "name": "gn_weight",
      "shape": [
        8192
      ]
    },
    {
      "dtype": "float32",
      "name": "gn_bias",
      "shape": [
        8192
      ]
    },
    {
      "dtype": "float32",
      "name": "bias",
      "shape": [
        1,
        8192,
        1,
        1
      ]
    }
  ],
  "outputs": [
    {
      "dtype": "float32",
      "shape": [
        1,
        8192,
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
    "linear_bias",
    "gn_weight",
    "gn_bias",
    "bias"
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
        "value": 512
      }
    },
    {
      "kind": "Assign",
      "targets": [
        {
          "id": "eps",
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
          "id": "linear_bias",
          "kind": "Name"
        }
      }
    },
    {
      "kind": "Assign",
      "targets": [
        {
          "elts": [
            {
              "id": "N",
              "kind": "Name"
            },
            {
              "id": "C",
              "kind": "Name"
            }
          ],
          "kind": "Tuple"
        }
      ],
      "value": {
        "attr": "shape",
        "kind": "Attribute",
        "value": {
          "id": "x",
          "kind": "Name"
        }
      }
    },
    {
      "kind": "Assign",
      "targets": [
        {
          "id": "G",
          "kind": "Name"
        }
      ],
      "value": {
        "id": "num_groups",
        "kind": "Name"
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
            "id": "N",
            "kind": "Name"
          },
          {
            "id": "G",
            "kind": "Name"
          },
          {
            "kind": "BinOp",
            "left": {
              "id": "C",
              "kind": "Name"
            },
            "op": {
              "kind": "FloorDiv"
            },
            "right": {
              "id": "G",
              "kind": "Name"
            }
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
              "kind": null,
              "value": 2
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
              "kind": null,
              "value": 2
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
                "id": "eps",
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
        "args": [
          {
            "id": "N",
            "kind": "Name"
          },
          {
            "id": "C",
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
          "attr": "min",
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
          "id": "x",
          "kind": "Name"
        }
      ],
      "value": {
        "args": [
          {
            "kind": null,
            "value": 1
          },
          {
            "kind": null,
            "value": 1
          },
          {
            "id": "N",
            "kind": "Name"
          },
          {
            "kind": null,
            "value": 1
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

Gemm + GroupNorm + Min + BiasAdd.

The input generator, concrete test values, baseline implementation, optimized reference, correctness tests, and grading logic are hidden. Do not use `interpret=True` or a plain-JAX fallback.
