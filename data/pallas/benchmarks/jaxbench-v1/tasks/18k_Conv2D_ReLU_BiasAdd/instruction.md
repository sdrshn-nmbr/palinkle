# 18k_Conv2D_ReLU_BiasAdd

Implement this pinned JAXBench workload as a normally lowered TPU Pallas kernel. Preserve the original workload semantics and original deployment shapes. The hidden verifier creates the exact inputs, computes the JAX baseline, checks normal Pallas lowering, captures a TPU profile, and compares performance against XLA.

1_Conv2D_ReLU_BiasAdd — JAXBench fused operator workload.

## Configuration

```json
{
  "batch_size": 128,
  "in_channels": 64,
  "kernel_size": 3,
  "name": "1_Conv2D_ReLU_BiasAdd",
  "out_channels": 128
}
```

## Required interface

```python
def workload(x, weight, conv_bias, bias):
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
        128,
        64,
        128,
        128
      ]
    },
    {
      "dtype": "bfloat16",
      "name": "weight",
      "shape": [
        128,
        64,
        3,
        3
      ]
    },
    {
      "dtype": "bfloat16",
      "name": "conv_bias",
      "shape": [
        128
      ]
    },
    {
      "dtype": "bfloat16",
      "name": "bias",
      "shape": [
        128,
        1,
        1
      ]
    }
  ],
  "outputs": [
    {
      "dtype": "bfloat16",
      "shape": [
        128,
        128,
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
    "weight",
    "conv_bias",
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
            "id": "weight",
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
    "jax": "jax",
    "jnp": "jax.numpy"
  },
  "module_values": {},
  "unresolved_names": []
}
```

Conv2D + ReLU + BiasAdd.

The input generator, concrete test values, baseline implementation, optimized reference, correctness tests, and grading logic are hidden. Do not use `interpret=True` or a plain-JAX fallback.
