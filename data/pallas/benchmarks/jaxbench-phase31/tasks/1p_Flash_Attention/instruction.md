# 1p_Flash_Attention

Implement this pinned JAXBench workload as a normally lowered TPU Pallas kernel. Preserve the original workload semantics and original deployment shapes. The hidden verifier creates the exact inputs, computes the JAX baseline, checks normal Pallas lowering, captures a TPU profile, and compares performance against XLA.

Vanilla Multi-Head Causal Attention — Baseline (64 heads, seq=2048).

Standard scaled dot-product attention with causal mask.
No GQA, no softcap, no sliding window — pure MHA baseline.
Matches Pallas flash_attention kernel config.

## Configuration

```json
{
  "batch": 4,
  "head_dim": 128,
  "model": "Baseline-MHA",
  "name": "flash_attention_baseline",
  "num_heads": 64,
  "operator": "causal_mha",
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
        64,
        4096,
        128
      ]
    },
    {
      "dtype": "bfloat16",
      "name": "key",
      "shape": [
        4,
        64,
        4096,
        128
      ]
    },
    {
      "dtype": "bfloat16",
      "name": "value",
      "shape": [
        4,
        64,
        4096,
        128
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
              "id": "H",
              "kind": "Name"
            },
            {
              "id": "S",
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
              "id": "query",
              "kind": "Name"
            },
            {
              "id": "key",
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
          "id": "output",
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
            "id": "value",
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
        "id": "output",
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

Standard causal multi-head attention: QK^T -> mask -> softmax -> AV.

The input generator, concrete test values, baseline implementation, optimized reference, correctness tests, and grading logic are hidden. Do not use `interpret=True` or a plain-JAX fallback.
