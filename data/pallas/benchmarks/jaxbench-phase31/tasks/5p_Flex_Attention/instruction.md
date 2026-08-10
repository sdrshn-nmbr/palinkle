# 5p_Flex_Attention

Implement this pinned JAXBench workload as a normally lowered TPU Pallas kernel. Preserve the original workload semantics and original deployment shapes. The hidden verifier creates the exact inputs, computes the JAX baseline, checks normal Pallas lowering, captures a TPU profile, and compares performance against XLA.

Flex Attention — Llama-3.1-70B with custom score modification.

Flexible attention with arbitrary score_mod function support, the JAX
equivalent of PyTorch's flex_attention. Baseline uses causal mask with
relative position bias as the score modifier.
From MaxText layers/attention_op.py (dot_product_attention with masks).

## Configuration

```json
{
  "batch": 4,
  "head_dim": 128,
  "model": "Llama-3.1-70B",
  "name": "llama3_70b_flex_attention",
  "num_heads": 64,
  "operator": "flex_attention",
  "seq_len": 4096
}
```

## Required interface

```python
def workload(q, k, v, rel_pos_bias):
    ...
```

## Tensor contract

```json
{
  "inputs": [
    {
      "dtype": "bfloat16",
      "name": "q",
      "shape": [
        4,
        64,
        4096,
        128
      ]
    },
    {
      "dtype": "bfloat16",
      "name": "k",
      "shape": [
        4,
        64,
        4096,
        128
      ]
    },
    {
      "dtype": "bfloat16",
      "name": "v",
      "shape": [
        4,
        64,
        4096,
        128
      ]
    },
    {
      "dtype": "bfloat16",
      "name": "rel_pos_bias",
      "shape": [
        64,
        4096,
        4096
      ]
    }
  ],
  "outputs": [
    {
      "dtype": "bfloat16",
      "shape": [
        4,
        64,
        4096,
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
    "q",
    "k",
    "v",
    "rel_pos_bias"
  ],
  "body": [
    {
      "kind": "Assign",
      "targets": [
        {
          "id": "D",
          "kind": "Name"
        }
      ],
      "value": {
        "kind": "Subscript",
        "slice": {
          "kind": null,
          "value": "head_dim"
        },
        "value": {
          "id": "CONFIG",
          "kind": "Name"
        }
      }
    },
    {
      "kind": "Assign",
      "targets": [
        {
          "id": "S",
          "kind": "Name"
        }
      ],
      "value": {
        "kind": "Subscript",
        "slice": {
          "kind": null,
          "value": "seq_len"
        },
        "value": {
          "id": "CONFIG",
          "kind": "Name"
        }
      }
    },
    {
      "kind": "Assign",
      "targets": [
        {
          "id": "sm_scale",
          "kind": "Name"
        }
      ],
      "value": {
        "kind": "BinOp",
        "left": {
          "id": "D",
          "kind": "Name"
        },
        "op": {
          "kind": "Pow"
        },
        "right": {
          "kind": "UnaryOp",
          "op": {
            "kind": "USub"
          },
          "operand": {
            "kind": null,
            "value": 0.5
          }
        }
      }
    },
    {
      "kind": "Assign",
      "targets": [
        {
          "id": "attn",
          "kind": "Name"
        }
      ],
      "value": {
        "kind": "BinOp",
        "left": {
          "args": [
            {
              "kind": null,
              "value": "bhqd,bhkd->bhqk"
            },
            {
              "id": "q",
              "kind": "Name"
            },
            {
              "id": "k",
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
        },
        "op": {
          "kind": "Mult"
        },
        "right": {
          "id": "sm_scale",
          "kind": "Name"
        }
      }
    },
    {
      "kind": "Assign",
      "targets": [
        {
          "id": "attn",
          "kind": "Name"
        }
      ],
      "value": {
        "kind": "BinOp",
        "left": {
          "id": "attn",
          "kind": "Name"
        },
        "op": {
          "kind": "Add"
        },
        "right": {
          "kind": "Subscript",
          "slice": {
            "elts": [
              {
                "kind": null,
                "value": null
              },
              {
                "kind": "Slice",
                "lower": null,
                "step": null,
                "upper": null
              },
              {
                "kind": "Slice",
                "lower": null,
                "step": null,
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
            "id": "rel_pos_bias",
            "kind": "Name"
          }
        }
      }
    },
    {
      "kind": "Assign",
      "targets": [
        {
          "id": "causal",
          "kind": "Name"
        }
      ],
      "value": {
        "args": [
          {
            "args": [
              {
                "elts": [
                  {
                    "id": "S",
                    "kind": "Name"
                  },
                  {
                    "id": "S",
                    "kind": "Name"
                  }
                ],
                "kind": "Tuple"
              }
            ],
            "func": {
              "attr": "ones",
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
                  "attr": "bool_",
                  "kind": "Attribute",
                  "value": {
                    "id": "jnp",
                    "kind": "Name"
                  }
                }
              }
            ],
            "kind": "Call"
          }
        ],
        "func": {
          "attr": "tril",
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
          "id": "attn",
          "kind": "Name"
        }
      ],
      "value": {
        "args": [
          {
            "kind": "Subscript",
            "slice": {
              "elts": [
                {
                  "kind": null,
                  "value": null
                },
                {
                  "kind": null,
                  "value": null
                },
                {
                  "kind": "Slice",
                  "lower": null,
                  "step": null,
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
              "id": "causal",
              "kind": "Name"
            }
          },
          {
            "id": "attn",
            "kind": "Name"
          },
          {
            "kind": "UnaryOp",
            "op": {
              "kind": "USub"
            },
            "operand": {
              "kind": null,
              "value": 1e+30
            }
          }
        ],
        "func": {
          "attr": "where",
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
          "id": "attn",
          "kind": "Name"
        }
      ],
      "value": {
        "args": [
          {
            "id": "attn",
            "kind": "Name"
          }
        ],
        "func": {
          "attr": "softmax",
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
          }
        ],
        "kind": "Call"
      }
    },
    {
      "kind": "Assign",
      "targets": [
        {
          "id": "out",
          "kind": "Name"
        }
      ],
      "value": {
        "args": [
          {
            "kind": null,
            "value": "bhqk,bhkd->bhqd"
          },
          {
            "id": "attn",
            "kind": "Name"
          },
          {
            "id": "v",
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
      "kind": "Return",
      "value": {
        "id": "out",
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

Flex attention: dot-product attention with score modification.

The input generator, concrete test values, baseline implementation, optimized reference, correctness tests, and grading logic are hidden. Do not use `interpret=True` or a plain-JAX fallback.
