# 2p_GQA_Attention

Implement this pinned JAXBench workload as a normally lowered TPU Pallas kernel. Preserve the original workload semantics and original deployment shapes. The hidden verifier creates the exact inputs, computes the JAX baseline, checks normal Pallas lowering, captures a TPU profile, and compares performance against XLA.

Grouped Query Attention (GQA) — Llama 3.1 405B. Extracted from MaxText.

## Configuration

```json
{
  "batch": 4,
  "emb_dim": 16384,
  "head_dim": 128,
  "model": "Llama-3.1-405B",
  "name": "llama3_405b_gqa",
  "num_kv_heads": 8,
  "num_query_heads": 128,
  "operator": "gqa_attention",
  "seq_len": 4096
}
```

## Required interface

```python
def workload(query, key, value):
    ...
```

## Tensor contract

```json
{
  "inputs": [
    {
      "dtype": "bfloat16",
      "name": "query",
      "shape": [
        4,
        4096,
        128,
        128
      ]
    },
    {
      "dtype": "bfloat16",
      "name": "key",
      "shape": [
        4,
        4096,
        8,
        128
      ]
    },
    {
      "dtype": "bfloat16",
      "name": "value",
      "shape": [
        4,
        4096,
        8,
        128
      ]
    }
  ],
  "outputs": [
    {
      "dtype": "bfloat16",
      "shape": [
        4,
        4096,
        128,
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
    "query",
    "key",
    "value"
  ],
  "body": [
    {
      "kind": "Assign",
      "targets": [
        {
          "elts": [
            {
              "id": "B",
              "kind": "Name"
            },
            {
              "id": "S",
              "kind": "Name"
            },
            {
              "id": "Hq",
              "kind": "Name"
            },
            {
              "id": "D",
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
          "id": "query",
          "kind": "Name"
        }
      }
    },
    {
      "kind": "Assign",
      "targets": [
        {
          "id": "Hkv",
          "kind": "Name"
        }
      ],
      "value": {
        "kind": "Subscript",
        "slice": {
          "kind": null,
          "value": 2
        },
        "value": {
          "attr": "shape",
          "kind": "Attribute",
          "value": {
            "id": "key",
            "kind": "Name"
          }
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
        "kind": "BinOp",
        "left": {
          "id": "Hq",
          "kind": "Name"
        },
        "op": {
          "kind": "FloorDiv"
        },
        "right": {
          "id": "Hkv",
          "kind": "Name"
        }
      }
    },
    {
      "kind": "Assign",
      "targets": [
        {
          "id": "key",
          "kind": "Name"
        }
      ],
      "value": {
        "args": [
          {
            "id": "B",
            "kind": "Name"
          },
          {
            "id": "S",
            "kind": "Name"
          },
          {
            "id": "Hq",
            "kind": "Name"
          },
          {
            "id": "D",
            "kind": "Name"
          }
        ],
        "func": {
          "attr": "reshape",
          "kind": "Attribute",
          "value": {
            "args": [
              {
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
                      "kind": null,
                      "value": null
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
                  "id": "key",
                  "kind": "Name"
                }
              },
              {
                "id": "G",
                "kind": "Name"
              }
            ],
            "func": {
              "attr": "repeat",
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
                  "value": 3
                }
              }
            ],
            "kind": "Call"
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
          "id": "value",
          "kind": "Name"
        }
      ],
      "value": {
        "args": [
          {
            "id": "B",
            "kind": "Name"
          },
          {
            "id": "S",
            "kind": "Name"
          },
          {
            "id": "Hq",
            "kind": "Name"
          },
          {
            "id": "D",
            "kind": "Name"
          }
        ],
        "func": {
          "attr": "reshape",
          "kind": "Attribute",
          "value": {
            "args": [
              {
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
                      "kind": null,
                      "value": null
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
                  "id": "value",
                  "kind": "Name"
                }
              },
              {
                "id": "G",
                "kind": "Name"
              }
            ],
            "func": {
              "attr": "repeat",
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
                  "value": 3
                }
              }
            ],
            "kind": "Call"
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
          "id": "q",
          "kind": "Name"
        }
      ],
      "value": {
        "args": [
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
            "value": 1
          },
          {
            "kind": null,
            "value": 3
          }
        ],
        "func": {
          "attr": "transpose",
          "kind": "Attribute",
          "value": {
            "id": "query",
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
          "id": "k",
          "kind": "Name"
        }
      ],
      "value": {
        "args": [
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
            "value": 1
          },
          {
            "kind": null,
            "value": 3
          }
        ],
        "func": {
          "attr": "transpose",
          "kind": "Attribute",
          "value": {
            "id": "key",
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
          "id": "v",
          "kind": "Name"
        }
      ],
      "value": {
        "args": [
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
            "value": 1
          },
          {
            "kind": null,
            "value": 3
          }
        ],
        "func": {
          "attr": "transpose",
          "kind": "Attribute",
          "value": {
            "id": "value",
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
          "id": "scale",
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
          "id": "scale",
          "kind": "Name"
        }
      }
    },
    {
      "kind": "Assign",
      "targets": [
        {
          "id": "mask",
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
            "keywords": [],
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
            "id": "mask",
            "kind": "Name"
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
              "value": 1000000000.0
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
        "args": [
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
            "value": 1
          },
          {
            "kind": null,
            "value": 3
          }
        ],
        "func": {
          "attr": "transpose",
          "kind": "Attribute",
          "value": {
            "id": "out",
            "kind": "Name"
          }
        },
        "keywords": [],
        "kind": "Call"
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

GQA attention: expand KV heads, scaled dot-product with causal mask.

The input generator, concrete test values, baseline implementation, optimized reference, correctness tests, and grading logic are hidden. Do not use `interpret=True` or a plain-JAX fallback.
