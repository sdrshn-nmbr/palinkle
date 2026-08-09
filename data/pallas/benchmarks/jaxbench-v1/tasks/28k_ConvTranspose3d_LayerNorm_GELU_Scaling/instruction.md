# 28k_ConvTranspose3d_LayerNorm_GELU_Scaling

Implement this pinned JAXBench workload as a normally lowered TPU Pallas kernel. Preserve the original workload semantics and original deployment shapes. The hidden verifier creates the exact inputs, computes the JAX baseline, checks normal Pallas lowering, captures a TPU profile, and compares performance against XLA.

34_ConvTranspose3d_LayerNorm_GELU_Scaling — JAXBench fused operator workload.

## Configuration

```json
{
  "batch_size": 32,
  "bias": true,
  "eps": 1e-05,
  "in_channels": 32,
  "kernel_size": 4,
  "name": "34_ConvTranspose3d_LayerNorm_GELU_Scaling",
  "out_channels": 64,
  "padding": 1,
  "scaling_factor": 1.0,
  "stride": 2
}
```

## Required interface

```python
def workload(x, conv_weight, conv_bias, ln_weight, ln_bias):
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
        32,
        32,
        16,
        32,
        32
      ]
    },
    {
      "dtype": "float32",
      "name": "conv_weight",
      "shape": [
        32,
        64,
        4,
        4,
        4
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
      "name": "ln_weight",
      "shape": [
        64
      ]
    },
    {
      "dtype": "float32",
      "name": "ln_bias",
      "shape": [
        64
      ]
    }
  ],
  "outputs": [
    {
      "dtype": "float32",
      "shape": [
        32,
        64,
        32,
        64,
        64
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
    "ln_weight",
    "ln_bias"
  ],
  "body": [
    {
      "kind": "Assign",
      "targets": [
        {
          "id": "stride",
          "kind": "Name"
        }
      ],
      "value": {
        "kind": null,
        "value": 2
      }
    },
    {
      "kind": "Assign",
      "targets": [
        {
          "id": "padding",
          "kind": "Name"
        }
      ],
      "value": {
        "kind": null,
        "value": 1
      }
    },
    {
      "kind": "Assign",
      "targets": [
        {
          "id": "kernel_size",
          "kind": "Name"
        }
      ],
      "value": {
        "kind": null,
        "value": 4
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
          "id": "scaling_factor",
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
                "value": 4
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
                "value": 4
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
          "id": "kernel",
          "kind": "Name"
        }
      ],
      "value": {
        "args": [
          {
            "id": "kernel",
            "kind": "Name"
          }
        ],
        "func": {
          "attr": "flip",
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
                  "value": 0
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
          }
        ],
        "kind": "Call"
      }
    },
    {
      "kind": "Assign",
      "targets": [
        {
          "elts": [
            {
              "id": "batch_size",
              "kind": "Name"
            },
            {
              "id": "d_in",
              "kind": "Name"
            },
            {
              "id": "h_in",
              "kind": "Name"
            },
            {
              "id": "w_in",
              "kind": "Name"
            },
            {
              "id": "channels",
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
          "id": "k",
          "kind": "Name"
        }
      ],
      "value": {
        "id": "kernel_size",
        "kind": "Name"
      }
    },
    {
      "kind": "Assign",
      "targets": [
        {
          "id": "d_dilated",
          "kind": "Name"
        }
      ],
      "value": {
        "kind": "BinOp",
        "left": {
          "id": "d_in",
          "kind": "Name"
        },
        "op": {
          "kind": "Add"
        },
        "right": {
          "kind": "BinOp",
          "left": {
            "kind": "BinOp",
            "left": {
              "id": "d_in",
              "kind": "Name"
            },
            "op": {
              "kind": "Sub"
            },
            "right": {
              "kind": null,
              "value": 1
            }
          },
          "op": {
            "kind": "Mult"
          },
          "right": {
            "kind": "BinOp",
            "left": {
              "id": "stride",
              "kind": "Name"
            },
            "op": {
              "kind": "Sub"
            },
            "right": {
              "kind": null,
              "value": 1
            }
          }
        }
      }
    },
    {
      "kind": "Assign",
      "targets": [
        {
          "id": "h_dilated",
          "kind": "Name"
        }
      ],
      "value": {
        "kind": "BinOp",
        "left": {
          "id": "h_in",
          "kind": "Name"
        },
        "op": {
          "kind": "Add"
        },
        "right": {
          "kind": "BinOp",
          "left": {
            "kind": "BinOp",
            "left": {
              "id": "h_in",
              "kind": "Name"
            },
            "op": {
              "kind": "Sub"
            },
            "right": {
              "kind": null,
              "value": 1
            }
          },
          "op": {
            "kind": "Mult"
          },
          "right": {
            "kind": "BinOp",
            "left": {
              "id": "stride",
              "kind": "Name"
            },
            "op": {
              "kind": "Sub"
            },
            "right": {
              "kind": null,
              "value": 1
            }
          }
        }
      }
    },
    {
      "kind": "Assign",
      "targets": [
        {
          "id": "w_dilated",
          "kind": "Name"
        }
      ],
      "value": {
        "kind": "BinOp",
        "left": {
          "id": "w_in",
          "kind": "Name"
        },
        "op": {
          "kind": "Add"
        },
        "right": {
          "kind": "BinOp",
          "left": {
            "kind": "BinOp",
            "left": {
              "id": "w_in",
              "kind": "Name"
            },
            "op": {
              "kind": "Sub"
            },
            "right": {
              "kind": null,
              "value": 1
            }
          },
          "op": {
            "kind": "Mult"
          },
          "right": {
            "kind": "BinOp",
            "left": {
              "id": "stride",
              "kind": "Name"
            },
            "op": {
              "kind": "Sub"
            },
            "right": {
              "kind": null,
              "value": 1
            }
          }
        }
      }
    },
    {
      "kind": "Assign",
      "targets": [
        {
          "id": "x_dilated",
          "kind": "Name"
        }
      ],
      "value": {
        "args": [
          {
            "elts": [
              {
                "id": "batch_size",
                "kind": "Name"
              },
              {
                "id": "d_dilated",
                "kind": "Name"
              },
              {
                "id": "h_dilated",
                "kind": "Name"
              },
              {
                "id": "w_dilated",
                "kind": "Name"
              },
              {
                "id": "channels",
                "kind": "Name"
              }
            ],
            "kind": "Tuple"
          }
        ],
        "func": {
          "attr": "zeros",
          "kind": "Attribute",
          "value": {
            "id": "jnp",
            "kind": "Name"
          }
        },
        "keywords": [
          {
            "arg": "dtype",
            "kind": "keyword",
            "value": {
              "attr": "dtype",
              "kind": "Attribute",
              "value": {
                "id": "x",
                "kind": "Name"
              }
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
          "id": "x_dilated",
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
          "attr": "set",
          "kind": "Attribute",
          "value": {
            "kind": "Subscript",
            "slice": {
              "elts": [
                {
                  "kind": "Slice",
                  "lower": null,
                  "step": null,
                  "upper": null
                },
                {
                  "kind": "Slice",
                  "lower": null,
                  "step": {
                    "id": "stride",
                    "kind": "Name"
                  },
                  "upper": null
                },
                {
                  "kind": "Slice",
                  "lower": null,
                  "step": {
                    "id": "stride",
                    "kind": "Name"
                  },
                  "upper": null
                },
                {
                  "kind": "Slice",
                  "lower": null,
                  "step": {
                    "id": "stride",
                    "kind": "Name"
                  },
                  "upper": null
                },
                {
                  "kind": "Slice",
                  "lower": null,
                  "step": null,
                  "upper": null
                }
              ],
              "kind": "Tuple"
            },
            "value": {
              "attr": "at",
              "kind": "Attribute",
              "value": {
                "id": "x_dilated",
                "kind": "Name"
              }
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
        "id": "x_dilated",
        "kind": "Name"
      }
    },
    {
      "kind": "Assign",
      "targets": [
        {
          "id": "pad",
          "kind": "Name"
        }
      ],
      "value": {
        "kind": "BinOp",
        "left": {
          "kind": "BinOp",
          "left": {
            "id": "k",
            "kind": "Name"
          },
          "op": {
            "kind": "Sub"
          },
          "right": {
            "kind": null,
            "value": 1
          }
        },
        "op": {
          "kind": "Sub"
        },
        "right": {
          "id": "padding",
          "kind": "Name"
        }
      }
    },
    {
      "kind": "Assign",
      "targets": [
        {
          "id": "jax_padding",
          "kind": "Name"
        }
      ],
      "value": {
        "elts": [
          {
            "elts": [
              {
                "id": "pad",
                "kind": "Name"
              },
              {
                "id": "pad",
                "kind": "Name"
              }
            ],
            "kind": "Tuple"
          },
          {
            "elts": [
              {
                "id": "pad",
                "kind": "Name"
              },
              {
                "id": "pad",
                "kind": "Name"
              }
            ],
            "kind": "Tuple"
          },
          {
            "elts": [
              {
                "id": "pad",
                "kind": "Name"
              },
              {
                "id": "pad",
                "kind": "Name"
              }
            ],
            "kind": "Tuple"
          }
        ],
        "kind": "Tuple"
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
            "id": "kernel",
            "kind": "Name"
          }
        ],
        "func": {
          "attr": "conv_general_dilated",
          "kind": "Attribute",
          "value": {
            "id": "lax",
            "kind": "Name"
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
              "id": "jax_padding",
              "kind": "Name"
            }
          },
          {
            "arg": "dimension_numbers",
            "kind": "keyword",
            "value": {
              "elts": [
                {
                  "kind": null,
                  "value": "NDHWC"
                },
                {
                  "kind": null,
                  "value": "DHWOI"
                },
                {
                  "kind": null,
                  "value": "NDHWC"
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
            "elts": [
              {
                "kind": null,
                "value": 0
              },
              {
                "kind": null,
                "value": 4
              },
              {
                "kind": null,
                "value": 1
              },
              {
                "kind": null,
                "value": 2
              },
              {
                "kind": null,
                "value": 3
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
            "id": "ln_weight",
            "kind": "Name"
          }
        },
        "op": {
          "kind": "Add"
        },
        "right": {
          "id": "ln_bias",
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
        "kind": "BinOp",
        "left": {
          "id": "x",
          "kind": "Name"
        },
        "op": {
          "kind": "Mult"
        },
        "right": {
          "id": "scaling_factor",
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
    "jnp": "jax.numpy",
    "lax": "jax.lax"
  },
  "module_values": {},
  "unresolved_names": []
}
```

ConvTranspose3d + LayerNorm + GELU + Scaling.

The input generator, concrete test values, baseline implementation, optimized reference, correctness tests, and grading logic are hidden. Do not use `interpret=True` or a plain-JAX fallback.
