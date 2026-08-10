# 6p_Paged_Attention

Implement this pinned JAXBench workload as a normally lowered TPU Pallas kernel. Preserve the original workload semantics and original deployment shapes. The hidden verifier creates the exact inputs, computes the JAX baseline, checks normal Pallas lowering, captures a TPU profile, and compares performance against XLA.

Ragged Paged Attention — Llama-3.1-70B inference decode.

Paged KV-cache attention with variable-length sequences, as used in
serving frameworks. Supports grouped-query attention (GQA).
From MaxText inference/paged_attention_kernel_v2.py.

## Configuration

```json
{
  "head_dim": 128,
  "max_seq_len": 4096,
  "model": "Llama-3.1-70B",
  "name": "llama3_70b_paged_attention",
  "num_kv_heads": 8,
  "num_query_heads": 64,
  "num_seqs": 64,
  "operator": "paged_attention",
  "page_size": 16,
  "pages_per_seq": 256
}
```

## Required interface

```python
def workload(queries, k_pages, v_pages, kv_lens, page_indices, cu_q_lens):
    ...
```

## Tensor contract

```json
{
  "inputs": [
    {
      "dtype": "bfloat16",
      "name": "queries",
      "shape": [
        64,
        64,
        128
      ]
    },
    {
      "dtype": "bfloat16",
      "name": "k_pages",
      "shape": [
        16384,
        16,
        8,
        128
      ]
    },
    {
      "dtype": "bfloat16",
      "name": "v_pages",
      "shape": [
        16384,
        16,
        8,
        128
      ]
    },
    {
      "dtype": "int32",
      "name": "kv_lens",
      "shape": [
        64
      ]
    },
    {
      "dtype": "int32",
      "name": "page_indices",
      "shape": [
        64,
        256
      ]
    },
    {
      "dtype": "int32",
      "name": "cu_q_lens",
      "shape": [
        65
      ]
    }
  ],
  "outputs": [
    {
      "dtype": "bfloat16",
      "shape": [
        64,
        64,
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
    "queries",
    "k_pages",
    "v_pages",
    "kv_lens",
    "page_indices",
    "cu_q_lens"
  ],
  "body": [
    {
      "kind": "Assign",
      "targets": [
        {
          "id": "num_seqs",
          "kind": "Name"
        }
      ],
      "value": {
        "kind": "Subscript",
        "slice": {
          "kind": null,
          "value": "num_seqs"
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
          "id": "num_q_heads",
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
          "id": "num_kv_heads",
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
          "id": "head_dim",
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
          "id": "page_size",
          "kind": "Name"
        }
      ],
      "value": {
        "kind": "Subscript",
        "slice": {
          "kind": null,
          "value": "page_size"
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
          "id": "max_seq_len",
          "kind": "Name"
        }
      ],
      "value": {
        "kind": "BinOp",
        "left": {
          "kind": "Subscript",
          "slice": {
            "kind": null,
            "value": "pages_per_seq"
          },
          "value": {
            "id": "CONFIG",
            "kind": "Name"
          }
        },
        "op": {
          "kind": "Mult"
        },
        "right": {
          "id": "page_size",
          "kind": "Name"
        }
      }
    },
    {
      "kind": "Assign",
      "targets": [
        {
          "id": "pages_per_seq",
          "kind": "Name"
        }
      ],
      "value": {
        "kind": "Subscript",
        "slice": {
          "kind": null,
          "value": "pages_per_seq"
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
          "id": "num_q_heads",
          "kind": "Name"
        },
        "op": {
          "kind": "FloorDiv"
        },
        "right": {
          "id": "num_kv_heads",
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
          "id": "head_dim",
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
      "args": {
        "args": [
          {
            "annotation": null,
            "arg": "seq_idx",
            "kind": "arg"
          }
        ],
        "defaults": [],
        "kind": "arguments",
        "kw_defaults": [],
        "kwarg": null,
        "kwonlyargs": [],
        "posonlyargs": [],
        "vararg": null
      },
      "body": [
        {
          "kind": "Assign",
          "targets": [
            {
              "id": "q_start",
              "kind": "Name"
            }
          ],
          "value": {
            "kind": "Subscript",
            "slice": {
              "id": "seq_idx",
              "kind": "Name"
            },
            "value": {
              "id": "cu_q_lens",
              "kind": "Name"
            }
          }
        },
        {
          "kind": "Assign",
          "targets": [
            {
              "id": "q_end",
              "kind": "Name"
            }
          ],
          "value": {
            "kind": "Subscript",
            "slice": {
              "kind": "BinOp",
              "left": {
                "id": "seq_idx",
                "kind": "Name"
              },
              "op": {
                "kind": "Add"
              },
              "right": {
                "kind": null,
                "value": 1
              }
            },
            "value": {
              "id": "cu_q_lens",
              "kind": "Name"
            }
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
                "id": "queries",
                "kind": "Name"
              },
              {
                "elts": [
                  {
                    "id": "q_start",
                    "kind": "Name"
                  },
                  {
                    "kind": null,
                    "value": 0
                  },
                  {
                    "kind": null,
                    "value": 0
                  }
                ],
                "kind": "Tuple"
              },
              {
                "elts": [
                  {
                    "kind": null,
                    "value": 1
                  },
                  {
                    "id": "num_q_heads",
                    "kind": "Name"
                  },
                  {
                    "id": "head_dim",
                    "kind": "Name"
                  }
                ],
                "kind": "Tuple"
              }
            ],
            "func": {
              "attr": "dynamic_slice",
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
              "id": "seq_pages",
              "kind": "Name"
            }
          ],
          "value": {
            "kind": "Subscript",
            "slice": {
              "id": "seq_idx",
              "kind": "Name"
            },
            "value": {
              "id": "page_indices",
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
                "id": "max_seq_len",
                "kind": "Name"
              },
              {
                "id": "num_kv_heads",
                "kind": "Name"
              },
              {
                "id": "head_dim",
                "kind": "Name"
              }
            ],
            "func": {
              "attr": "reshape",
              "kind": "Attribute",
              "value": {
                "kind": "Subscript",
                "slice": {
                  "id": "seq_pages",
                  "kind": "Name"
                },
                "value": {
                  "id": "k_pages",
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
              "id": "v",
              "kind": "Name"
            }
          ],
          "value": {
            "args": [
              {
                "id": "max_seq_len",
                "kind": "Name"
              },
              {
                "id": "num_kv_heads",
                "kind": "Name"
              },
              {
                "id": "head_dim",
                "kind": "Name"
              }
            ],
            "func": {
              "attr": "reshape",
              "kind": "Attribute",
              "value": {
                "kind": "Subscript",
                "slice": {
                  "id": "seq_pages",
                  "kind": "Name"
                },
                "value": {
                  "id": "v_pages",
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
                  "value": 1
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
                  "value": 1
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
            "kind": "BinOp",
            "left": {
              "args": [
                {
                  "kind": null,
                  "value": "qhd,khd->hqk"
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
              "id": "kv_len",
              "kind": "Name"
            }
          ],
          "value": {
            "kind": "Subscript",
            "slice": {
              "id": "seq_idx",
              "kind": "Name"
            },
            "value": {
              "id": "kv_lens",
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
            "comparators": [
              {
                "id": "kv_len",
                "kind": "Name"
              }
            ],
            "kind": "Compare",
            "left": {
              "args": [
                {
                  "id": "max_seq_len",
                  "kind": "Name"
                }
              ],
              "func": {
                "attr": "arange",
                "kind": "Attribute",
                "value": {
                  "id": "jnp",
                  "kind": "Name"
                }
              },
              "keywords": [],
              "kind": "Call"
            },
            "ops": [
              {
                "kind": "Lt"
              }
            ]
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
                    }
                  ],
                  "kind": "Tuple"
                },
                "value": {
                  "id": "mask",
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
                "value": "hqk,khd->qhd"
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
              }
            ],
            "func": {
              "attr": "squeeze",
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
      "decorator_list": [],
      "kind": "FunctionDef",
      "name": "attend_one_seq",
      "returns": null,
      "type_params": []
    },
    {
      "kind": "Assign",
      "targets": [
        {
          "id": "outputs",
          "kind": "Name"
        }
      ],
      "value": {
        "args": [
          {
            "args": [
              {
                "id": "num_seqs",
                "kind": "Name"
              }
            ],
            "func": {
              "attr": "arange",
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
          "args": [
            {
              "id": "attend_one_seq",
              "kind": "Name"
            }
          ],
          "func": {
            "attr": "vmap",
            "kind": "Attribute",
            "value": {
              "id": "jax",
              "kind": "Name"
            }
          },
          "keywords": [],
          "kind": "Call"
        },
        "keywords": [],
        "kind": "Call"
      }
    },
    {
      "kind": "Return",
      "value": {
        "id": "outputs",
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

Ragged paged attention: gather pages, compute GQA attention per sequence.

The input generator, concrete test values, baseline implementation, optimized reference, correctness tests, and grading logic are hidden. Do not use `interpret=True` or a plain-JAX fallback.
