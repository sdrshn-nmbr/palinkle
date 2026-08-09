# 17p_Triangle_Multiplication

Implement this pinned JAXBench workload as a normally lowered TPU Pallas kernel. Preserve the original workload semantics and original deployment shapes. The hidden verifier creates the exact inputs, computes the JAX baseline, checks normal Pallas lowering, captures a TPU profile, and compares performance against XLA.

Triangle Multiplicative Update (Outgoing) — AlphaFold2 768.

Contracts over the second residue index with gated projections and
layer normalization. Core structural operation in AlphaFold2.
From openxla/tokamax triangle_mult benchmarks.

## Configuration

```json
{
  "C": 128,
  "N": 1536,
  "direction": "outgoing",
  "model": "AlphaFold2",
  "name": "alphafold_768_triangle_mult",
  "operator": "triangle_mult_outgoing"
}
```

## Required interface

```python
def workload(pair_act, mask, left_proj_w, right_proj_w, left_gate_w, right_gate_w, center_scale, out_proj_w, out_gate_w):
    ...
```

## Tensor contract

```json
{
  "inputs": [
    {
      "dtype": "bfloat16",
      "name": "pair_act",
      "shape": [
        1536,
        1536,
        128
      ]
    },
    {
      "dtype": "bfloat16",
      "name": "mask",
      "shape": [
        1536,
        1536,
        1
      ]
    },
    {
      "dtype": "bfloat16",
      "name": "left_proj_w",
      "shape": [
        128,
        128
      ]
    },
    {
      "dtype": "bfloat16",
      "name": "right_proj_w",
      "shape": [
        128,
        128
      ]
    },
    {
      "dtype": "bfloat16",
      "name": "left_gate_w",
      "shape": [
        128,
        128
      ]
    },
    {
      "dtype": "bfloat16",
      "name": "right_gate_w",
      "shape": [
        128,
        128
      ]
    },
    {
      "dtype": "bfloat16",
      "name": "center_scale",
      "shape": [
        128
      ]
    },
    {
      "dtype": "bfloat16",
      "name": "out_proj_w",
      "shape": [
        128,
        128
      ]
    },
    {
      "dtype": "bfloat16",
      "name": "out_gate_w",
      "shape": [
        128,
        128
      ]
    }
  ],
  "outputs": [
    {
      "dtype": "bfloat16",
      "shape": [
        1536,
        1536,
        128
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
    "pair_act",
    "mask",
    "left_proj_w",
    "right_proj_w",
    "left_gate_w",
    "right_gate_w",
    "center_scale",
    "out_proj_w",
    "out_gate_w"
  ],
  "body": [
    {
      "kind": "Assign",
      "targets": [
        {
          "id": "act",
          "kind": "Name"
        }
      ],
      "value": {
        "kind": "BinOp",
        "left": {
          "id": "pair_act",
          "kind": "Name"
        },
        "op": {
          "kind": "Mult"
        },
        "right": {
          "id": "mask",
          "kind": "Name"
        }
      }
    },
    {
      "kind": "Assign",
      "targets": [
        {
          "id": "left_proj",
          "kind": "Name"
        }
      ],
      "value": {
        "args": [
          {
            "id": "act",
            "kind": "Name"
          },
          {
            "id": "left_proj_w",
            "kind": "Name"
          }
        ],
        "func": {
          "attr": "dot",
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
          "id": "right_proj",
          "kind": "Name"
        }
      ],
      "value": {
        "args": [
          {
            "id": "act",
            "kind": "Name"
          },
          {
            "id": "right_proj_w",
            "kind": "Name"
          }
        ],
        "func": {
          "attr": "dot",
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
          "id": "left_gate",
          "kind": "Name"
        }
      ],
      "value": {
        "args": [
          {
            "args": [
              {
                "id": "act",
                "kind": "Name"
              },
              {
                "id": "left_gate_w",
                "kind": "Name"
              }
            ],
            "func": {
              "attr": "dot",
              "kind": "Attribute",
              "value": {
                "id": "jnp",
                "kind": "Name"
              }
            },
            "keywords": [],
            "kind": "Call"
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
    },
    {
      "kind": "Assign",
      "targets": [
        {
          "id": "right_gate",
          "kind": "Name"
        }
      ],
      "value": {
        "args": [
          {
            "args": [
              {
                "id": "act",
                "kind": "Name"
              },
              {
                "id": "right_gate_w",
                "kind": "Name"
              }
            ],
            "func": {
              "attr": "dot",
              "kind": "Attribute",
              "value": {
                "id": "jnp",
                "kind": "Name"
              }
            },
            "keywords": [],
            "kind": "Call"
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
    },
    {
      "kind": "Assign",
      "targets": [
        {
          "id": "left_proj",
          "kind": "Name"
        }
      ],
      "value": {
        "kind": "BinOp",
        "left": {
          "id": "left_proj",
          "kind": "Name"
        },
        "op": {
          "kind": "Mult"
        },
        "right": {
          "id": "left_gate",
          "kind": "Name"
        }
      }
    },
    {
      "kind": "Assign",
      "targets": [
        {
          "id": "right_proj",
          "kind": "Name"
        }
      ],
      "value": {
        "kind": "BinOp",
        "left": {
          "id": "right_proj",
          "kind": "Name"
        },
        "op": {
          "kind": "Mult"
        },
        "right": {
          "id": "right_gate",
          "kind": "Name"
        }
      }
    },
    {
      "kind": "Assign",
      "targets": [
        {
          "id": "result",
          "kind": "Name"
        }
      ],
      "value": {
        "args": [
          {
            "kind": null,
            "value": "ikc,jkc->ijc"
          },
          {
            "id": "left_proj",
            "kind": "Name"
          },
          {
            "id": "right_proj",
            "kind": "Name"
          }
        ],
        "func": {
          "attr": "einsum",
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
          "id": "eps",
          "kind": "Name"
        }
      ],
      "value": {
        "kind": null,
        "value": 1e-06
      }
    },
    {
      "kind": "Assign",
      "targets": [
        {
          "id": "rms",
          "kind": "Name"
        }
      ],
      "value": {
        "args": [
          {
            "kind": "BinOp",
            "left": {
              "args": [
                {
                  "kind": "BinOp",
                  "left": {
                    "id": "result",
                    "kind": "Name"
                  },
                  "op": {
                    "kind": "Mult"
                  },
                  "right": {
                    "id": "result",
                    "kind": "Name"
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
    {
      "kind": "Assign",
      "targets": [
        {
          "id": "result",
          "kind": "Name"
        }
      ],
      "value": {
        "kind": "BinOp",
        "left": {
          "kind": "BinOp",
          "left": {
            "id": "result",
            "kind": "Name"
          },
          "op": {
            "kind": "Div"
          },
          "right": {
            "id": "rms",
            "kind": "Name"
          }
        },
        "op": {
          "kind": "Mult"
        },
        "right": {
          "id": "center_scale",
          "kind": "Name"
        }
      }
    },
    {
      "kind": "Assign",
      "targets": [
        {
          "id": "output",
          "kind": "Name"
        }
      ],
      "value": {
        "args": [
          {
            "id": "result",
            "kind": "Name"
          },
          {
            "id": "out_proj_w",
            "kind": "Name"
          }
        ],
        "func": {
          "attr": "dot",
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
          "id": "gate",
          "kind": "Name"
        }
      ],
      "value": {
        "args": [
          {
            "args": [
              {
                "id": "pair_act",
                "kind": "Name"
              },
              {
                "id": "out_gate_w",
                "kind": "Name"
              }
            ],
            "func": {
              "attr": "dot",
              "kind": "Attribute",
              "value": {
                "id": "jnp",
                "kind": "Name"
              }
            },
            "keywords": [],
            "kind": "Call"
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
    },
    {
      "kind": "Return",
      "value": {
        "kind": "BinOp",
        "left": {
          "id": "output",
          "kind": "Name"
        },
        "op": {
          "kind": "Mult"
        },
        "right": {
          "id": "gate",
          "kind": "Name"
        }
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

Triangle multiplicative update (outgoing): contracts over the second residue index.

The input generator, concrete test values, baseline implementation, optimized reference, correctness tests, and grading logic are hidden. Do not use `interpret=True` or a plain-JAX fallback.
