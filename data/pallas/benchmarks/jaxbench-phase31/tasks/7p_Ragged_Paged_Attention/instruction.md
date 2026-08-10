# 7p_Ragged_Paged_Attention

Implement this pinned JAXBench workload as a normally lowered TPU Pallas kernel. Preserve the original workload semantics and original deployment shapes. The hidden verifier creates the exact inputs, computes the JAX baseline, checks normal Pallas lowering, captures a TPU profile, and compares performance against XLA.

Ragged Paged Attention — Llama-3.1-70B mixed prefill+decode.

Reference implementation with data-dependent slicing on per-sequence boundaries.
Processes each sequence independently with variable-length queries and paged KV cache.
From JAX experimental pallas ops (ref_ragged_paged_attention).

## Configuration

```json
{
  "head_dim": 128,
  "max_num_batched_tokens": 4096,
  "max_num_seqs": 64,
  "model": "Llama-3.1-70B",
  "name": "ragged_paged_attention_llama70b",
  "num_kv_heads": 8,
  "num_q_heads": 64,
  "operator": "ragged_paged_attention",
  "page_size": 16,
  "pages_per_seq": 256
}
```

## Required interface

```python
def workload(queries, kv_pages, kv_lens, page_indices, cu_q_lens, num_seqs):
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
        4096,
        64,
        128
      ]
    },
    {
      "dtype": "bfloat16",
      "name": "kv_pages",
      "shape": [
        16384,
        16,
        16,
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
    },
    {
      "dtype": "int32",
      "name": "num_seqs",
      "shape": [
        1
      ]
    }
  ],
  "outputs": [
    {
      "dtype": "bfloat16",
      "shape": [
        4096,
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
    "kv_pages",
    "kv_lens",
    "page_indices",
    "cu_q_lens",
    "num_seqs"
  ],
  "body": [
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
          "kind": null,
          "value": 1.0
        },
        "op": {
          "kind": "Div"
        },
        "right": {
          "args": [
            {
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
          ],
          "func": {
            "attr": "sqrt",
            "kind": "Attribute",
            "value": {
              "id": "math",
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
          "id": "mask_value",
          "kind": "Name"
        }
      ],
      "value": {
        "id": "DEFAULT_MASK_VALUE",
        "kind": "Name"
      }
    },
    {
      "kind": "Assign",
      "targets": [
        {
          "elts": [
            {
              "id": "_",
              "kind": "Name"
            },
            {
              "id": "_",
              "kind": "Name"
            },
            {
              "id": "num_combined_kv_heads",
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
      "value": {
        "attr": "shape",
        "kind": "Attribute",
        "value": {
          "id": "kv_pages",
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
        "kind": "BinOp",
        "left": {
          "id": "num_combined_kv_heads",
          "kind": "Name"
        },
        "op": {
          "kind": "FloorDiv"
        },
        "right": {
          "kind": null,
          "value": 2
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
          "value": 1
        },
        "value": {
          "attr": "shape",
          "kind": "Attribute",
          "value": {
            "id": "queries",
            "kind": "Name"
          }
        }
      }
    },
    {
      "kind": "Assign",
      "targets": [
        {
          "id": "num_query_per_kv",
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
          "id": "max_seqs",
          "kind": "Name"
        }
      ],
      "value": {
        "kind": "Subscript",
        "slice": {
          "kind": null,
          "value": "max_num_seqs"
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
          "id": "tokens_per_seq",
          "kind": "Name"
        }
      ],
      "value": {
        "kind": "BinOp",
        "left": {
          "kind": "Subscript",
          "slice": {
            "kind": null,
            "value": "max_num_batched_tokens"
          },
          "value": {
            "id": "CONFIG",
            "kind": "Name"
          }
        },
        "op": {
          "kind": "FloorDiv"
        },
        "right": {
          "id": "max_seqs",
          "kind": "Name"
        }
      }
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
        "elts": [],
        "kind": "List"
      }
    },
    {
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
              "id": "i",
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
              "id": "kv_len",
              "kind": "Name"
            }
          ],
          "value": {
            "kind": "Subscript",
            "slice": {
              "id": "i",
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
              "id": "indices",
              "kind": "Name"
            }
          ],
          "value": {
            "kind": "Subscript",
            "slice": {
              "id": "i",
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
                    "id": "tokens_per_seq",
                    "kind": "Name"
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
              "id": "k",
              "kind": "Name"
            }
          ],
          "value": {
            "args": [
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
                  "elts": [
                    {
                      "id": "indices",
                      "kind": "Name"
                    },
                    {
                      "kind": "Slice",
                      "lower": null,
                      "step": null,
                      "upper": null
                    },
                    {
                      "kind": "Slice",
                      "lower": {
                        "kind": null,
                        "value": 0
                      },
                      "step": {
                        "kind": null,
                        "value": 2
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
                  "id": "kv_pages",
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
                  "elts": [
                    {
                      "id": "indices",
                      "kind": "Name"
                    },
                    {
                      "kind": "Slice",
                      "lower": null,
                      "step": null,
                      "upper": null
                    },
                    {
                      "kind": "Slice",
                      "lower": {
                        "kind": null,
                        "value": 1
                      },
                      "step": {
                        "kind": null,
                        "value": 2
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
                  "id": "kv_pages",
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
                "id": "num_query_per_kv",
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
                "id": "num_query_per_kv",
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
            "keywords": [
              {
                "arg": "preferred_element_type",
                "kind": "keyword",
                "value": {
                  "attr": "float32",
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
        },
        {
          "kind": "AugAssign",
          "op": {
            "kind": "Mult"
          },
          "target": {
            "id": "attn",
            "kind": "Name"
          },
          "value": {
            "id": "sm_scale",
            "kind": "Name"
          }
        },
        {
          "kind": "Assign",
          "targets": [
            {
              "id": "q_span",
              "kind": "Name"
            }
          ],
          "value": {
            "kind": "BinOp",
            "left": {
              "kind": "BinOp",
              "left": {
                "id": "kv_len",
                "kind": "Name"
              },
              "op": {
                "kind": "Sub"
              },
              "right": {
                "id": "tokens_per_seq",
                "kind": "Name"
              }
            },
            "op": {
              "kind": "Add"
            },
            "right": {
              "args": [
                {
                  "attr": "int32",
                  "kind": "Attribute",
                  "value": {
                    "id": "jnp",
                    "kind": "Name"
                  }
                },
                {
                  "attr": "shape",
                  "kind": "Attribute",
                  "value": {
                    "id": "attn",
                    "kind": "Name"
                  }
                },
                {
                  "kind": null,
                  "value": 1
                }
              ],
              "func": {
                "attr": "broadcasted_iota",
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
          }
        },
        {
          "kind": "Assign",
          "targets": [
            {
              "id": "kv_span",
              "kind": "Name"
            }
          ],
          "value": {
            "args": [
              {
                "attr": "int32",
                "kind": "Attribute",
                "value": {
                  "id": "jnp",
                  "kind": "Name"
                }
              },
              {
                "attr": "shape",
                "kind": "Attribute",
                "value": {
                  "id": "attn",
                  "kind": "Name"
                }
              },
              {
                "kind": null,
                "value": 2
              }
            ],
            "func": {
              "attr": "broadcasted_iota",
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
              "id": "mask",
              "kind": "Name"
            }
          ],
          "value": {
            "kind": "BinOp",
            "left": {
              "comparators": [
                {
                  "id": "kv_span",
                  "kind": "Name"
                }
              ],
              "kind": "Compare",
              "left": {
                "id": "q_span",
                "kind": "Name"
              },
              "ops": [
                {
                  "kind": "Lt"
                }
              ]
            },
            "op": {
              "kind": "BitOr"
            },
            "right": {
              "comparators": [
                {
                  "id": "kv_len",
                  "kind": "Name"
                }
              ],
              "kind": "Compare",
              "left": {
                "id": "kv_span",
                "kind": "Name"
              },
              "ops": [
                {
                  "kind": "GtE"
                }
              ]
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
            "args": [
              {
                "id": "mask",
                "kind": "Name"
              },
              {
                "id": "mask_value",
                "kind": "Name"
              },
              {
                "id": "attn",
                "kind": "Name"
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
                "attr": "dtype",
                "kind": "Attribute",
                "value": {
                  "id": "v",
                  "kind": "Name"
                }
              }
            ],
            "func": {
              "attr": "astype",
              "kind": "Attribute",
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
            "keywords": [],
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
                "attr": "dtype",
                "kind": "Attribute",
                "value": {
                  "id": "queries",
                  "kind": "Name"
                }
              }
            ],
            "func": {
              "attr": "astype",
              "kind": "Attribute",
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
            "keywords": [],
            "kind": "Call"
          }
        },
        {
          "kind": "Assign",
          "targets": [
            {
              "id": "is_valid",
              "kind": "Name"
            }
          ],
          "value": {
            "comparators": [
              {
                "kind": "Subscript",
                "slice": {
                  "kind": null,
                  "value": 0
                },
                "value": {
                  "id": "num_seqs",
                  "kind": "Name"
                }
              }
            ],
            "kind": "Compare",
            "left": {
              "id": "i",
              "kind": "Name"
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
              "id": "out",
              "kind": "Name"
            }
          ],
          "value": {
            "args": [
              {
                "id": "is_valid",
                "kind": "Name"
              },
              {
                "id": "out",
                "kind": "Name"
              },
              {
                "kind": null,
                "value": 0.0
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
          "kind": "Expr",
          "value": {
            "args": [
              {
                "id": "out",
                "kind": "Name"
              }
            ],
            "func": {
              "attr": "append",
              "kind": "Attribute",
              "value": {
                "id": "outputs",
                "kind": "Name"
              }
            },
            "keywords": [],
            "kind": "Call"
          }
        }
      ],
      "iter": {
        "args": [
          {
            "id": "max_seqs",
            "kind": "Name"
          }
        ],
        "func": {
          "id": "range",
          "kind": "Name"
        },
        "keywords": [],
        "kind": "Call"
      },
      "kind": "For",
      "orelse": [],
      "target": {
        "id": "i",
        "kind": "Name"
      }
    },
    {
      "kind": "Return",
      "value": {
        "args": [
          {
            "id": "outputs",
            "kind": "Name"
          }
        ],
        "func": {
          "attr": "concatenate",
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
    }
  ],
  "format": "canonical_python_ast_semantics_v1",
  "helper_functions": {},
  "imports": {
    "jax": "jax",
    "jnp": "jax.numpy",
    "math": "math"
  },
  "module_values": {
    "DEFAULT_MASK_VALUE": {
      "kind": "BinOp",
      "left": {
        "kind": "UnaryOp",
        "op": {
          "kind": "USub"
        },
        "operand": {
          "kind": null,
          "value": 0.7
        }
      },
      "op": {
        "kind": "Mult"
      },
      "right": {
        "args": [
          {
            "attr": "max",
            "kind": "Attribute",
            "value": {
              "args": [
                {
                  "args": [
                    {
                      "kind": null,
                      "value": "float32"
                    }
                  ],
                  "func": {
                    "attr": "dtype",
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
                "attr": "finfo",
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
        ],
        "func": {
          "id": "float",
          "kind": "Name"
        },
        "keywords": [],
        "kind": "Call"
      }
    }
  },
  "unresolved_names": []
}
```

Ragged paged attention using static shapes and masking for JIT compatibility.

Processes each sequence independently, avoiding data-dependent slicing.

The input generator, concrete test values, baseline implementation, optimized reference, correctness tests, and grading logic are hidden. Do not use `interpret=True` or a plain-JAX fallback.
