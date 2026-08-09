# 35k_Gemm_Scaling_Hardtanh_GELU

Implement this pinned JAXBench workload as a normally lowered TPU Pallas kernel. Preserve the original workload semantics and original deployment shapes. The hidden verifier creates the exact inputs, computes the JAX baseline, checks normal Pallas lowering, captures a TPU profile, and compares performance against XLA.

53_Gemm_Scaling_Hardtanh_GELU — JAXBench fused operator workload.

## Configuration

```json
{
  "batch_size": 4096,
  "hardtanh_max": 2,
  "hardtanh_min": -2,
  "in_features": 8192,
  "name": "53_Gemm_Scaling_Hardtanh_GELU",
  "out_features": 8192,
  "scaling_factor": 0.5
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
          "value": 0.5
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
              "value": 2
            }
          },
          {
            "kind": null,
            "value": 2
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
            "kind": null,
            "value": 0.5
          }
        },
        "op": {
          "kind": "Mult"
        },
        "right": {
          "kind": "BinOp",
          "left": {
            "kind": null,
            "value": 1.0
          },
          "op": {
            "kind": "Add"
          },
          "right": {
            "args": [
              {
                "kind": "BinOp",
                "left": {
                  "args": [
                    {
                      "kind": "BinOp",
                      "left": {
                        "kind": null,
                        "value": 2.0
                      },
                      "op": {
                        "kind": "Div"
                      },
                      "right": {
                        "attr": "pi",
                        "kind": "Attribute",
                        "value": {
                          "id": "jnp",
                          "kind": "Name"
                        }
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
                },
                "op": {
                  "kind": "Mult"
                },
                "right": {
                  "kind": "BinOp",
                  "left": {
                    "id": "x",
                    "kind": "Name"
                  },
                  "op": {
                    "kind": "Add"
                  },
                  "right": {
                    "kind": "BinOp",
                    "left": {
                      "kind": null,
                      "value": 0.044715
                    },
                    "op": {
                      "kind": "Mult"
                    },
                    "right": {
                      "kind": "BinOp",
                      "left": {
                        "id": "x",
                        "kind": "Name"
                      },
                      "op": {
                        "kind": "Pow"
                      },
                      "right": {
                        "kind": null,
                        "value": 3
                      }
                    }
                  }
                }
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

Gemm + Scaling + Hardtanh + GELU.

The input generator, concrete test values, baseline implementation, optimized reference, correctness tests, and grading logic are hidden. Do not use `interpret=True` or a plain-JAX fallback.
