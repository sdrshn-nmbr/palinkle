# 3p_MLA_Attention

Implement this pinned JAXBench workload as a normally lowered TPU Pallas kernel. Preserve the original workload semantics and original deployment shapes. The hidden verifier creates the exact inputs, computes the JAX baseline, checks normal Pallas lowering, captures a TPU profile, and compares performance against XLA.

Multi-head Latent Attention (MLA) — DeepSeek V3 671B. Extracted from MaxText.

## Configuration

```json
{
  "batch": 4,
  "emb_dim": 7168,
  "kv_lora_rank": 512,
  "model": "DeepSeek-V3-671B",
  "name": "deepseek_v3_mla",
  "num_heads": 128,
  "operator": "mla_attention",
  "q_lora_rank": 1536,
  "qk_nope_head_dim": 128,
  "qk_rope_head_dim": 64,
  "rope_theta": 10000,
  "seq_len": 2048,
  "v_head_dim": 128
}
```

## Required interface

```python
def workload(x, q_down_proj, q_up_proj, kv_down_proj, k_up_proj, v_up_proj, o_proj):
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
        4,
        2048,
        7168
      ]
    },
    {
      "dtype": "bfloat16",
      "name": "q_down_proj",
      "shape": [
        7168,
        1536
      ]
    },
    {
      "dtype": "bfloat16",
      "name": "q_up_proj",
      "shape": [
        1536,
        24576
      ]
    },
    {
      "dtype": "bfloat16",
      "name": "kv_down_proj",
      "shape": [
        7168,
        576
      ]
    },
    {
      "dtype": "bfloat16",
      "name": "k_up_proj",
      "shape": [
        512,
        16384
      ]
    },
    {
      "dtype": "bfloat16",
      "name": "v_up_proj",
      "shape": [
        512,
        16384
      ]
    },
    {
      "dtype": "bfloat16",
      "name": "o_proj",
      "shape": [
        16384,
        7168
      ]
    }
  ],
  "outputs": [
    {
      "dtype": "bfloat16",
      "shape": [
        4,
        2048,
        7168
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
    "q_down_proj",
    "q_up_proj",
    "kv_down_proj",
    "k_up_proj",
    "v_up_proj",
    "o_proj"
  ],
  "body": [
    {
      "kind": "Assign",
      "targets": [
        {
          "id": "C",
          "kind": "Name"
        }
      ],
      "value": {
        "id": "CONFIG",
        "kind": "Name"
      }
    },
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
          "id": "H",
          "kind": "Name"
        }
      ],
      "value": {
        "kind": "Subscript",
        "slice": {
          "kind": null,
          "value": "num_heads"
        },
        "value": {
          "id": "C",
          "kind": "Name"
        }
      }
    },
    {
      "kind": "Assign",
      "targets": [
        {
          "elts": [
            {
              "id": "nope",
              "kind": "Name"
            },
            {
              "id": "rope",
              "kind": "Name"
            },
            {
              "id": "vd",
              "kind": "Name"
            }
          ],
          "kind": "Tuple"
        }
      ],
      "value": {
        "elts": [
          {
            "kind": "Subscript",
            "slice": {
              "kind": null,
              "value": "qk_nope_head_dim"
            },
            "value": {
              "id": "C",
              "kind": "Name"
            }
          },
          {
            "kind": "Subscript",
            "slice": {
              "kind": null,
              "value": "qk_rope_head_dim"
            },
            "value": {
              "id": "C",
              "kind": "Name"
            }
          },
          {
            "kind": "Subscript",
            "slice": {
              "kind": null,
              "value": "v_head_dim"
            },
            "value": {
              "id": "C",
              "kind": "Name"
            }
          }
        ],
        "kind": "Tuple"
      }
    },
    {
      "kind": "Assign",
      "targets": [
        {
          "id": "kvl",
          "kind": "Name"
        }
      ],
      "value": {
        "kind": "Subscript",
        "slice": {
          "kind": null,
          "value": "kv_lora_rank"
        },
        "value": {
          "id": "C",
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
            "args": [
              {
                "id": "x",
                "kind": "Name"
              },
              {
                "id": "q_down_proj",
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
          },
          {
            "id": "q_up_proj",
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
          "id": "q",
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
            "id": "H",
            "kind": "Name"
          },
          {
            "kind": "BinOp",
            "left": {
              "id": "nope",
              "kind": "Name"
            },
            "op": {
              "kind": "Add"
            },
            "right": {
              "id": "rope",
              "kind": "Name"
            }
          }
        ],
        "func": {
          "attr": "reshape",
          "kind": "Attribute",
          "value": {
            "id": "q",
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
              "id": "q_nope",
              "kind": "Name"
            },
            {
              "id": "q_rope",
              "kind": "Name"
            }
          ],
          "kind": "Tuple"
        }
      ],
      "value": {
        "elts": [
          {
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
                  "kind": "Slice",
                  "lower": null,
                  "step": null,
                  "upper": {
                    "id": "nope",
                    "kind": "Name"
                  }
                }
              ],
              "kind": "Tuple"
            },
            "value": {
              "id": "q",
              "kind": "Name"
            }
          },
          {
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
                  "kind": "Slice",
                  "lower": {
                    "id": "nope",
                    "kind": "Name"
                  },
                  "step": null,
                  "upper": null
                }
              ],
              "kind": "Tuple"
            },
            "value": {
              "id": "q",
              "kind": "Name"
            }
          }
        ],
        "kind": "Tuple"
      }
    },
    {
      "kind": "Assign",
      "targets": [
        {
          "id": "kv",
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
            "id": "kv_down_proj",
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
              "id": "k_latent",
              "kind": "Name"
            },
            {
              "id": "k_rope_raw",
              "kind": "Name"
            }
          ],
          "kind": "Tuple"
        }
      ],
      "value": {
        "elts": [
          {
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
                  "kind": "Slice",
                  "lower": null,
                  "step": null,
                  "upper": {
                    "id": "kvl",
                    "kind": "Name"
                  }
                }
              ],
              "kind": "Tuple"
            },
            "value": {
              "id": "kv",
              "kind": "Name"
            }
          },
          {
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
                  "kind": "Slice",
                  "lower": {
                    "id": "kvl",
                    "kind": "Name"
                  },
                  "step": null,
                  "upper": null
                }
              ],
              "kind": "Tuple"
            },
            "value": {
              "id": "kv",
              "kind": "Name"
            }
          }
        ],
        "kind": "Tuple"
      }
    },
    {
      "kind": "Assign",
      "targets": [
        {
          "id": "k_nope",
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
            "id": "H",
            "kind": "Name"
          },
          {
            "id": "nope",
            "kind": "Name"
          }
        ],
        "func": {
          "attr": "reshape",
          "kind": "Attribute",
          "value": {
            "args": [
              {
                "id": "k_latent",
                "kind": "Name"
              },
              {
                "id": "k_up_proj",
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
              "id": "cos",
              "kind": "Name"
            },
            {
              "id": "sin",
              "kind": "Name"
            }
          ],
          "kind": "Tuple"
        }
      ],
      "value": {
        "args": [
          {
            "id": "rope",
            "kind": "Name"
          },
          {
            "id": "S",
            "kind": "Name"
          },
          {
            "kind": "Subscript",
            "slice": {
              "kind": null,
              "value": "rope_theta"
            },
            "value": {
              "id": "C",
              "kind": "Name"
            }
          },
          {
            "attr": "dtype",
            "kind": "Attribute",
            "value": {
              "id": "x",
              "kind": "Name"
            }
          }
        ],
        "func": {
          "id": "_compute_rope",
          "kind": "Name"
        },
        "keywords": [],
        "kind": "Call"
      }
    },
    {
      "kind": "Assign",
      "targets": [
        {
          "id": "k_rope",
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
              "id": "k_rope_raw",
              "kind": "Name"
            }
          },
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
                "id": "H",
                "kind": "Name"
              },
              {
                "id": "rope",
                "kind": "Name"
              }
            ],
            "kind": "Tuple"
          }
        ],
        "func": {
          "attr": "broadcast_to",
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
          "id": "q_rope",
          "kind": "Name"
        }
      ],
      "value": {
        "args": [
          {
            "id": "q_rope",
            "kind": "Name"
          },
          {
            "id": "cos",
            "kind": "Name"
          },
          {
            "id": "sin",
            "kind": "Name"
          }
        ],
        "func": {
          "id": "_apply_rope",
          "kind": "Name"
        },
        "keywords": [],
        "kind": "Call"
      }
    },
    {
      "kind": "Assign",
      "targets": [
        {
          "id": "k_rope",
          "kind": "Name"
        }
      ],
      "value": {
        "args": [
          {
            "id": "k_rope",
            "kind": "Name"
          },
          {
            "id": "cos",
            "kind": "Name"
          },
          {
            "id": "sin",
            "kind": "Name"
          }
        ],
        "func": {
          "id": "_apply_rope",
          "kind": "Name"
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
            "id": "B",
            "kind": "Name"
          },
          {
            "id": "S",
            "kind": "Name"
          },
          {
            "id": "H",
            "kind": "Name"
          },
          {
            "id": "vd",
            "kind": "Name"
          }
        ],
        "func": {
          "attr": "reshape",
          "kind": "Attribute",
          "value": {
            "args": [
              {
                "id": "k_latent",
                "kind": "Name"
              },
              {
                "id": "v_up_proj",
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
        "keywords": [],
        "kind": "Call"
      }
    },
    {
      "kind": "Assign",
      "targets": [
        {
          "id": "q_full",
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
            "args": [
              {
                "elts": [
                  {
                    "id": "q_nope",
                    "kind": "Name"
                  },
                  {
                    "id": "q_rope",
                    "kind": "Name"
                  }
                ],
                "kind": "List"
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
          "id": "k_full",
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
            "args": [
              {
                "elts": [
                  {
                    "id": "k_nope",
                    "kind": "Name"
                  },
                  {
                    "id": "k_rope",
                    "kind": "Name"
                  }
                ],
                "kind": "List"
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
            "id": "v",
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
          "id": "hd",
          "kind": "Name"
        }
      ],
      "value": {
        "kind": "BinOp",
        "left": {
          "id": "nope",
          "kind": "Name"
        },
        "op": {
          "kind": "Add"
        },
        "right": {
          "id": "rope",
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
          "args": [
            {
              "kind": null,
              "value": "bhqd,bhkd->bhqk"
            },
            {
              "id": "q_full",
              "kind": "Name"
            },
            {
              "id": "k_full",
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
          "kind": "BinOp",
          "left": {
            "id": "hd",
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
            "id": "B",
            "kind": "Name"
          },
          {
            "id": "S",
            "kind": "Name"
          },
          {
            "kind": "BinOp",
            "left": {
              "id": "H",
              "kind": "Name"
            },
            "op": {
              "kind": "Mult"
            },
            "right": {
              "id": "vd",
              "kind": "Name"
            }
          }
        ],
        "func": {
          "attr": "reshape",
          "kind": "Attribute",
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
            "id": "out",
            "kind": "Name"
          },
          {
            "id": "o_proj",
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
    }
  ],
  "format": "canonical_python_ast_semantics_v1",
  "helper_functions": {
    "_apply_rope": {
      "arguments": [
        "x",
        "cos",
        "sin"
      ],
      "body": [
        {
          "kind": "Assign",
          "targets": [
            {
              "elts": [
                {
                  "id": "x1",
                  "kind": "Name"
                },
                {
                  "id": "x2",
                  "kind": "Name"
                }
              ],
              "kind": "Tuple"
            }
          ],
          "value": {
            "elts": [
              {
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
                      "kind": "Slice",
                      "lower": null,
                      "step": {
                        "kind": null,
                        "value": 2
                      },
                      "upper": null
                    }
                  ],
                  "kind": "Tuple"
                },
                "value": {
                  "id": "x",
                  "kind": "Name"
                }
              },
              {
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
                    }
                  ],
                  "kind": "Tuple"
                },
                "value": {
                  "id": "x",
                  "kind": "Name"
                }
              }
            ],
            "kind": "Tuple"
          }
        },
        {
          "kind": "Assign",
          "targets": [
            {
              "id": "cos",
              "kind": "Name"
            }
          ],
          "value": {
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
              "id": "cos",
              "kind": "Name"
            }
          }
        },
        {
          "kind": "Assign",
          "targets": [
            {
              "id": "sin",
              "kind": "Name"
            }
          ],
          "value": {
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
              "id": "sin",
              "kind": "Name"
            }
          }
        },
        {
          "kind": "Assign",
          "targets": [
            {
              "id": "rotated",
              "kind": "Name"
            }
          ],
          "value": {
            "args": [
              {
                "elts": [
                  {
                    "kind": "BinOp",
                    "left": {
                      "kind": "BinOp",
                      "left": {
                        "id": "x1",
                        "kind": "Name"
                      },
                      "op": {
                        "kind": "Mult"
                      },
                      "right": {
                        "id": "cos",
                        "kind": "Name"
                      }
                    },
                    "op": {
                      "kind": "Sub"
                    },
                    "right": {
                      "kind": "BinOp",
                      "left": {
                        "id": "x2",
                        "kind": "Name"
                      },
                      "op": {
                        "kind": "Mult"
                      },
                      "right": {
                        "id": "sin",
                        "kind": "Name"
                      }
                    }
                  },
                  {
                    "kind": "BinOp",
                    "left": {
                      "kind": "BinOp",
                      "left": {
                        "id": "x1",
                        "kind": "Name"
                      },
                      "op": {
                        "kind": "Mult"
                      },
                      "right": {
                        "id": "sin",
                        "kind": "Name"
                      }
                    },
                    "op": {
                      "kind": "Add"
                    },
                    "right": {
                      "kind": "BinOp",
                      "left": {
                        "id": "x2",
                        "kind": "Name"
                      },
                      "op": {
                        "kind": "Mult"
                      },
                      "right": {
                        "id": "cos",
                        "kind": "Name"
                      }
                    }
                  }
                ],
                "kind": "List"
              }
            ],
            "func": {
              "attr": "stack",
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
              }
            ],
            "kind": "Call"
          }
        },
        {
          "kind": "Return",
          "value": {
            "args": [
              {
                "attr": "shape",
                "kind": "Attribute",
                "value": {
                  "id": "x",
                  "kind": "Name"
                }
              }
            ],
            "func": {
              "attr": "reshape",
              "kind": "Attribute",
              "value": {
                "id": "rotated",
                "kind": "Name"
              }
            },
            "keywords": [],
            "kind": "Call"
          }
        }
      ]
    },
    "_compute_rope": {
      "arguments": [
        "head_dim",
        "seq_len",
        "theta",
        "dtype"
      ],
      "body": [
        {
          "kind": "Assign",
          "targets": [
            {
              "id": "freqs",
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
              "kind": "BinOp",
              "left": {
                "id": "theta",
                "kind": "Name"
              },
              "op": {
                "kind": "Pow"
              },
              "right": {
                "kind": "BinOp",
                "left": {
                  "args": [
                    {
                      "kind": null,
                      "value": 0
                    },
                    {
                      "id": "head_dim",
                      "kind": "Name"
                    },
                    {
                      "kind": null,
                      "value": 2
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
                  "keywords": [
                    {
                      "arg": "dtype",
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
                },
                "op": {
                  "kind": "Div"
                },
                "right": {
                  "id": "head_dim",
                  "kind": "Name"
                }
              }
            }
          }
        },
        {
          "kind": "Assign",
          "targets": [
            {
              "id": "pos",
              "kind": "Name"
            }
          ],
          "value": {
            "args": [
              {
                "id": "seq_len",
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
            "keywords": [
              {
                "arg": "dtype",
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
          "kind": "Assign",
          "targets": [
            {
              "id": "angles",
              "kind": "Name"
            }
          ],
          "value": {
            "args": [
              {
                "id": "pos",
                "kind": "Name"
              },
              {
                "id": "freqs",
                "kind": "Name"
              }
            ],
            "func": {
              "attr": "outer",
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
            "elts": [
              {
                "args": [
                  {
                    "id": "dtype",
                    "kind": "Name"
                  }
                ],
                "func": {
                  "attr": "astype",
                  "kind": "Attribute",
                  "value": {
                    "args": [
                      {
                        "id": "angles",
                        "kind": "Name"
                      }
                    ],
                    "func": {
                      "attr": "cos",
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
              },
              {
                "args": [
                  {
                    "id": "dtype",
                    "kind": "Name"
                  }
                ],
                "func": {
                  "attr": "astype",
                  "kind": "Attribute",
                  "value": {
                    "args": [
                      {
                        "id": "angles",
                        "kind": "Name"
                      }
                    ],
                    "func": {
                      "attr": "sin",
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
            ],
            "kind": "Tuple"
          }
        }
      ]
    }
  },
  "imports": {
    "jax": "jax",
    "jnp": "jax.numpy"
  },
  "module_values": {},
  "unresolved_names": []
}
```

MLA: low-rank KV compression with separated position/content attention.

The input generator, concrete test values, baseline implementation, optimized reference, correctness tests, and grading logic are hidden. Do not use `interpret=True` or a plain-JAX fallback.
