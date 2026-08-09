# 4p_Sparse_Attention

Implement this pinned JAXBench workload as a normally lowered TPU Pallas kernel. Preserve the original workload semantics and original deployment shapes. The hidden verifier creates the exact inputs, computes the JAX baseline, checks normal Pallas lowering, captures a TPU profile, and compares performance against XLA.

Sparse (Splash) Attention — Llama-3.1-70B GQA with causal mask.

Vanilla JAX baseline for sparse/splash attention: standard dot-product
attention with causal masking and grouped-query attention (GQA).
From MaxText kernels/attention/splash_attention_kernel.py.

## Configuration

```json
{
  "batch": 4,
  "head_dim": 128,
  "model": "Llama-3.1-70B",
  "name": "llama3_70b_sparse_attention",
  "num_kv_heads": 8,
  "num_query_heads": 64,
  "operator": "sparse_attention",
  "seq_len": 4096
}
```

## Required interface

```python
def workload(q, k, v):
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
        64,
        4096,
        128
      ]
    },
    {
      "dtype": "bfloat16",
      "name": "k",
      "shape": [
        8,
        4096,
        128
      ]
    },
    {
      "dtype": "bfloat16",
      "name": "v",
      "shape": [
        8,
        4096,
        128
      ]
    }
  ],
  "outputs": [
    {
      "dtype": "bfloat16",
      "shape": [
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
    "v"
  ],
  "body": [
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
          "id": "H_q",
          "kind": "Name"
        }
      ],
      "value": {
        "kind": "Subscript",
        "slice": {
          "kind": null,
          "value": "num_query_heads"
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
          "id": "H_kv",
          "kind": "Name"
        }
      ],
      "value": {
        "kind": "Subscript",
        "slice": {
          "kind": null,
          "value": "num_kv_heads"
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
          "id": "num_q_per_kv",
          "kind": "Name"
        }
      ],
      "value": {
        "kind": "BinOp",
        "left": {
          "id": "H_q",
          "kind": "Name"
        },
        "op": {
          "kind": "FloorDiv"
        },
        "right": {
          "id": "H_kv",
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
        "args": [
          {
            "id": "k",
            "kind": "Name"
          },
          {
            "id": "num_q_per_kv",
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
              "value": 0
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
          "id": "v",
          "kind": "Name"
        }
      ],
      "value": {
        "args": [
          {
            "id": "v",
            "kind": "Name"
          },
          {
            "id": "num_q_per_kv",
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
              "value": 0
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
          "id": "attn",
          "kind": "Name"
        }
      ],
      "value": {
        "args": [
          {
            "kind": null,
            "value": "hqd,hkd->hqk"
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
            "value": "hqk,hkd->hqd"
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

Causal GQA attention: splash attention baseline.

The input generator, concrete test values, baseline implementation, optimized reference, correctness tests, and grading logic are hidden. Do not use `interpret=True` or a plain-JAX fallback.
