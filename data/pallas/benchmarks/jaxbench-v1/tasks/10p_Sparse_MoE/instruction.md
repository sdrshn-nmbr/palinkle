# 10p_Sparse_MoE

Implement this pinned JAXBench workload as a normally lowered TPU Pallas kernel. Preserve the original workload semantics and original deployment shapes. The hidden verifier creates the exact inputs, computes the JAX baseline, checks normal Pallas lowering, captures a TPU profile, and compares performance against XLA.

Sparse Mixture of Experts (MoE) — Mixtral 8x7B. Extracted from MaxText.

## Configuration

```json
{
  "batch": 2,
  "emb_dim": 4096,
  "mlp_dim": 14336,
  "model": "Mixtral-8x7B",
  "name": "mixtral_8x7b_moe",
  "num_experts": 8,
  "num_experts_per_tok": 2,
  "operator": "sparse_moe",
  "seq_len": 4096
}
```

## Required interface

```python
def workload(x, router_weights, expert_gate_kernels, expert_up_kernels, expert_down_kernels):
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
        2,
        4096,
        4096
      ]
    },
    {
      "dtype": "bfloat16",
      "name": "router_weights",
      "shape": [
        4096,
        8
      ]
    },
    {
      "dtype": "bfloat16",
      "name": "expert_gate_kernels",
      "shape": [
        8,
        4096,
        14336
      ]
    },
    {
      "dtype": "bfloat16",
      "name": "expert_up_kernels",
      "shape": [
        8,
        4096,
        14336
      ]
    },
    {
      "dtype": "bfloat16",
      "name": "expert_down_kernels",
      "shape": [
        8,
        14336,
        4096
      ]
    }
  ],
  "outputs": [
    {
      "dtype": "float32",
      "shape": [
        2,
        4096,
        4096
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
    "router_weights",
    "expert_gate_kernels",
    "expert_up_kernels",
    "expert_down_kernels"
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
              "id": "E",
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
          "id": "N",
          "kind": "Name"
        }
      ],
      "value": {
        "kind": "Subscript",
        "slice": {
          "kind": "UnaryOp",
          "op": {
            "kind": "USub"
          },
          "operand": {
            "kind": null,
            "value": 1
          }
        },
        "value": {
          "attr": "shape",
          "kind": "Attribute",
          "value": {
            "id": "router_weights",
            "kind": "Name"
          }
        }
      }
    },
    {
      "kind": "Assign",
      "targets": [
        {
          "id": "K",
          "kind": "Name"
        }
      ],
      "value": {
        "kind": "Subscript",
        "slice": {
          "kind": null,
          "value": "num_experts_per_tok"
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
          "id": "logits",
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
            "id": "router_weights",
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
          "elts": [
            {
              "id": "top_k_logits",
              "kind": "Name"
            },
            {
              "id": "top_k_indices",
              "kind": "Name"
            }
          ],
          "kind": "Tuple"
        }
      ],
      "value": {
        "args": [
          {
            "id": "logits",
            "kind": "Name"
          },
          {
            "id": "K",
            "kind": "Name"
          }
        ],
        "func": {
          "attr": "top_k",
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
        "keywords": [],
        "kind": "Call"
      }
    },
    {
      "kind": "Assign",
      "targets": [
        {
          "id": "router_probs",
          "kind": "Name"
        }
      ],
      "value": {
        "args": [
          {
            "id": "top_k_logits",
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
          "id": "gate_out",
          "kind": "Name"
        }
      ],
      "value": {
        "args": [
          {
            "args": [
              {
                "kind": null,
                "value": "bse,nem->bsnm"
              },
              {
                "id": "x",
                "kind": "Name"
              },
              {
                "id": "expert_gate_kernels",
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
        ],
        "func": {
          "attr": "silu",
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
          "id": "up_out",
          "kind": "Name"
        }
      ],
      "value": {
        "args": [
          {
            "kind": null,
            "value": "bse,nem->bsnm"
          },
          {
            "id": "x",
            "kind": "Name"
          },
          {
            "id": "expert_up_kernels",
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
          "id": "hidden",
          "kind": "Name"
        }
      ],
      "value": {
        "kind": "BinOp",
        "left": {
          "id": "gate_out",
          "kind": "Name"
        },
        "op": {
          "kind": "Mult"
        },
        "right": {
          "id": "up_out",
          "kind": "Name"
        }
      }
    },
    {
      "kind": "Assign",
      "targets": [
        {
          "id": "expert_outputs",
          "kind": "Name"
        }
      ],
      "value": {
        "args": [
          {
            "kind": null,
            "value": "bsnm,nme->bsne"
          },
          {
            "id": "hidden",
            "kind": "Name"
          },
          {
            "id": "expert_down_kernels",
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
          "id": "one_hot",
          "kind": "Name"
        }
      ],
      "value": {
        "args": [
          {
            "id": "top_k_indices",
            "kind": "Name"
          },
          {
            "id": "N",
            "kind": "Name"
          }
        ],
        "func": {
          "attr": "one_hot",
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
          "id": "weighted",
          "kind": "Name"
        }
      ],
      "value": {
        "kind": "BinOp",
        "left": {
          "id": "one_hot",
          "kind": "Name"
        },
        "op": {
          "kind": "Mult"
        },
        "right": {
          "kind": "Subscript",
          "slice": {
            "elts": [
              {
                "kind": null,
                "value": {
                  "kind": "Ellipsis"
                }
              },
              {
                "kind": null,
                "value": null
              }
            ],
            "kind": "Tuple"
          },
          "value": {
            "id": "router_probs",
            "kind": "Name"
          }
        }
      }
    },
    {
      "kind": "Assign",
      "targets": [
        {
          "id": "expert_weights",
          "kind": "Name"
        }
      ],
      "value": {
        "args": [],
        "func": {
          "attr": "sum",
          "kind": "Attribute",
          "value": {
            "id": "weighted",
            "kind": "Name"
          }
        },
        "keywords": [
          {
            "arg": "axis",
            "kind": "keyword",
            "value": {
              "kind": null,
              "value": 2
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
            "value": "bsne,bsn->bse"
          },
          {
            "id": "expert_outputs",
            "kind": "Name"
          },
          {
            "id": "expert_weights",
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

Sparse MoE with einsum-based batched expert computation.

The input generator, concrete test values, baseline implementation, optimized reference, correctness tests, and grading logic are hidden. Do not use `interpret=True` or a plain-JAX fallback.
