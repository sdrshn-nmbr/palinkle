# 46k_Conv2d_GroupNorm_Tanh_HardSwish_ResidualAdd_LogSumExp

Implement this pinned JAXBench workload as a normally lowered TPU Pallas kernel. Preserve the original workload semantics and original deployment shapes. The hidden verifier creates the exact inputs, computes the JAX baseline, checks normal Pallas lowering, captures a TPU profile, and compares performance against XLA.

92_Conv2d_GroupNorm_Tanh_HardSwish_ResidualAdd_LogSumExp — JAXBench fused operator workload.

## Configuration

```json
{
  "batch_size": 128,
  "groups": 16,
  "in_channels": 8,
  "kernel_size": 3,
  "name": "92_Conv2d_GroupNorm_Tanh_HardSwish_ResidualAdd_LogSumExp",
  "out_channels": 64
}
```

## Required interface

```python
def workload(x, conv_weight, conv_bias, gn_weight, gn_bias):
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
        128,
        8,
        128,
        128
      ]
    },
    {
      "dtype": "float32",
      "name": "conv_weight",
      "shape": [
        64,
        8,
        3,
        3
      ]
    },
    {
      "dtype": "float32",
      "name": "conv_bias",
      "shape": [
        64
      ]
    },
    {
      "dtype": "float32",
      "name": "gn_weight",
      "shape": [
        64
      ]
    },
    {
      "dtype": "float32",
      "name": "gn_bias",
      "shape": [
        64
      ]
    }
  ],
  "outputs": [
    {
      "dtype": "float32",
      "shape": [
        128,
        1,
        126,
        126
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
    "conv_weight",
    "conv_bias",
    "gn_weight",
    "gn_bias"
  ],
  "body": [
    {
      "kind": "Assign",
      "targets": [
        {
          "id": "groups",
          "kind": "Name"
        }
      ],
      "value": {
        "kind": null,
        "value": 16
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
          "id": "x_nhwc",
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
            "elts": [
              {
                "kind": null,
                "value": 0
              },
              {
                "kind": null,
                "value": 2
              },
              {
                "kind": null,
                "value": 3
              },
              {
                "kind": null,
                "value": 1
              }
            ],
            "kind": "Tuple"
          }
        ],
        "func": {
          "attr": "transpose",
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
          "id": "kernel",
          "kind": "Name"
        }
      ],
      "value": {
        "args": [
          {
            "id": "conv_weight",
            "kind": "Name"
          },
          {
            "elts": [
              {
                "kind": null,
                "value": 2
              },
              {
                "kind": null,
                "value": 3
              },
              {
                "kind": null,
                "value": 1
              },
              {
                "kind": null,
                "value": 0
              }
            ],
            "kind": "Tuple"
          }
        ],
        "func": {
          "attr": "transpose",
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
          "id": "x_conv",
          "kind": "Name"
        }
      ],
      "value": {
        "args": [
          {
            "id": "x_nhwc",
            "kind": "Name"
          },
          {
            "id": "kernel",
            "kind": "Name"
          }
        ],
        "func": {
          "attr": "conv_general_dilated",
          "kind": "Attribute",
          "value": {
            "attr": "lax",
            "kind": "Attribute",
            "value": {
              "id": "jax",
              "kind": "Name"
            }
          }
        },
        "keywords": [
          {
            "arg": "window_strides",
            "kind": "keyword",
            "value": {
              "elts": [
                {
                  "kind": null,
                  "value": 1
                },
                {
                  "kind": null,
                  "value": 1
                }
              ],
              "kind": "Tuple"
            }
          },
          {
            "arg": "padding",
            "kind": "keyword",
            "value": {
              "kind": null,
              "value": "VALID"
            }
          },
          {
            "arg": "dimension_numbers",
            "kind": "keyword",
            "value": {
              "elts": [
                {
                  "kind": null,
                  "value": "NHWC"
                },
                {
                  "kind": null,
                  "value": "HWIO"
                },
                {
                  "kind": null,
                  "value": "NHWC"
                }
              ],
              "kind": "Tuple"
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
          "id": "x_conv",
          "kind": "Name"
        }
      ],
      "value": {
        "kind": "BinOp",
        "left": {
          "id": "x_conv",
          "kind": "Name"
        },
        "op": {
          "kind": "Add"
        },
        "right": {
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
              "kind": null,
              "value": 1
            },
            {
              "kind": "UnaryOp",
              "op": {
                "kind": "USub"
              },
              "operand": {
                "kind": null,
                "value": 1
              }
            }
          ],
          "func": {
            "attr": "reshape",
            "kind": "Attribute",
            "value": {
              "id": "conv_bias",
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
          "id": "x_conv",
          "kind": "Name"
        }
      ],
      "value": {
        "args": [
          {
            "id": "x_conv",
            "kind": "Name"
          },
          {
            "elts": [
              {
                "kind": null,
                "value": 0
              },
              {
                "kind": null,
                "value": 3
              },
              {
                "kind": null,
                "value": 1
              },
              {
                "kind": null,
                "value": 2
              }
            ],
            "kind": "Tuple"
          }
        ],
        "func": {
          "attr": "transpose",
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
          "elts": [
            {
              "id": "N",
              "kind": "Name"
            },
            {
              "id": "C",
              "kind": "Name"
            },
            {
              "id": "H",
              "kind": "Name"
            },
            {
              "id": "W",
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
          "id": "x_conv",
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
            "id": "N",
            "kind": "Name"
          },
          {
            "id": "groups",
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
              "id": "groups",
              "kind": "Name"
            }
          },
          {
            "id": "H",
            "kind": "Name"
          },
          {
            "id": "W",
            "kind": "Name"
          }
        ],
        "func": {
          "attr": "reshape",
          "kind": "Attribute",
          "value": {
            "id": "x_conv",
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
              "elts": [
                {
                  "kind": null,
                  "value": 2
                },
                {
                  "kind": null,
                  "value": 3
                },
                {
                  "kind": null,
                  "value": 4
                }
              ],
              "kind": "Tuple"
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
              "elts": [
                {
                  "kind": null,
                  "value": 2
                },
                {
                  "kind": null,
                  "value": 3
                },
                {
                  "kind": null,
                  "value": 4
                }
              ],
              "kind": "Tuple"
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
          },
          {
            "id": "H",
            "kind": "Name"
          },
          {
            "id": "W",
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
          "id": "x_norm",
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
            "args": [
              {
                "kind": null,
                "value": 1
              },
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
                "kind": null,
                "value": 1
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
                "id": "gn_weight",
                "kind": "Name"
              }
            },
            "keywords": [],
            "kind": "Call"
          }
        },
        "op": {
          "kind": "Add"
        },
        "right": {
          "args": [
            {
              "kind": null,
              "value": 1
            },
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
              "kind": null,
              "value": 1
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
              "id": "gn_bias",
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
          "id": "x_tanh",
          "kind": "Name"
        }
      ],
      "value": {
        "args": [
          {
            "id": "x_norm",
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
    },
    {
      "kind": "Assign",
      "targets": [
        {
          "id": "x_hard_swish",
          "kind": "Name"
        }
      ],
      "value": {
        "kind": "BinOp",
        "left": {
          "kind": "BinOp",
          "left": {
            "id": "x_tanh",
            "kind": "Name"
          },
          "op": {
            "kind": "Mult"
          },
          "right": {
            "args": [
              {
                "args": [
                  {
                    "kind": "BinOp",
                    "left": {
                      "id": "x_tanh",
                      "kind": "Name"
                    },
                    "op": {
                      "kind": "Add"
                    },
                    "right": {
                      "kind": null,
                      "value": 3
                    }
                  },
                  {
                    "kind": null,
                    "value": 0
                  }
                ],
                "func": {
                  "attr": "maximum",
                  "kind": "Attribute",
                  "value": {
                    "id": "jnp",
                    "kind": "Name"
                  }
                },
                "keywords": [],
                "kind": "Call"
              },
              {
                "kind": null,
                "value": 6
              }
            ],
            "func": {
              "attr": "minimum",
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
          "kind": "Div"
        },
        "right": {
          "kind": null,
          "value": 6
        }
      }
    },
    {
      "kind": "Assign",
      "targets": [
        {
          "id": "x_res",
          "kind": "Name"
        }
      ],
      "value": {
        "kind": "BinOp",
        "left": {
          "id": "x_conv",
          "kind": "Name"
        },
        "op": {
          "kind": "Add"
        },
        "right": {
          "id": "x_hard_swish",
          "kind": "Name"
        }
      }
    },
    {
      "kind": "Assign",
      "targets": [
        {
          "id": "x_logsumexp",
          "kind": "Name"
        }
      ],
      "value": {
        "args": [
          {
            "id": "x_res",
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
      "kind": "Return",
      "value": {
        "id": "x_logsumexp",
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

Conv2d + GroupNorm + Tanh + HardSwish + ResidualAdd + LogSumExp.

The input generator, concrete test values, baseline implementation, optimized reference, correctness tests, and grading logic are hidden. Do not use `interpret=True` or a plain-JAX fallback.
