# 31k_Gemm_BatchNorm_GELU_ReLU

Implement this pinned JAXBench workload as a normally lowered TPU Pallas kernel. Preserve the original workload semantics and original deployment shapes. The hidden verifier creates the exact inputs, computes the JAX baseline, checks normal Pallas lowering, captures a TPU profile, and compares performance against XLA.

41_Gemm_BatchNorm_GELU_ReLU — JAXBench fused operator workload.

## Configuration

```json
{
  "batch_size": 16384,
  "in_features": 8192,
  "name": "41_Gemm_BatchNorm_GELU_ReLU",
  "out_features": 8192
}
```

## Required interface

```python
def workload(x, gemm_weight, gemm_bias, bn_weight, bn_bias):
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
        16384,
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
      "name": "bn_weight",
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
    }
  ],
  "outputs": [
    {
      "dtype": "bfloat16",
      "shape": [
        16384,
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
    "bn_weight",
    "bn_bias"
  ],
  "body": [
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
              "value": 0
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
              "kind": "Pow"
            },
            "right": {
              "kind": null,
              "value": 2
            }
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
              "value": 0
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
          },
          "op": {
            "kind": "Mult"
          },
          "right": {
            "id": "bn_weight",
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
          "attr": "relu",
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
    "jax": "jax",
    "jnp": "jax.numpy"
  },
  "module_values": {},
  "unresolved_names": []
}
```

Gemm + BatchNorm + GELU + ReLU.

The input generator, concrete test values, baseline implementation, optimized reference, correctness tests, and grading logic are hidden. Do not use `interpret=True` or a plain-JAX fallback.
