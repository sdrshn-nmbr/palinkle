# 16p_Mamba2_SSD

Implement this pinned JAXBench workload as a normally lowered TPU Pallas kernel. Preserve the original workload semantics and original deployment shapes. The hidden verifier creates the exact inputs, computes the JAX baseline, checks normal Pallas lowering, captures a TPU profile, and compares performance against XLA.

Mamba-2 State Space Duality (SSD) — Dao & Gu.

The SSD layer shows that structured state space models are equivalent to a form
of linear attention with input-dependent (selective) decay. This is the matrix
(parallel) form of Mamba-2's core computation.

Paper: "Transformers are SSMs" (Dao & Gu, 2024)
Mamba-2 is the dominant alternative to standard transformers in 2024-2025.

Config based on Mamba-2-2.7B from the paper.

## Configuration

```json
{
  "batch": 4,
  "d_model": 2560,
  "d_state": 128,
  "head_dim": 64,
  "model": "Mamba-2-2.7B",
  "name": "mamba2_2_7b_ssd",
  "num_heads": 64,
  "operator": "state_space_duality",
  "seq_len": 4096
}
```

## Required interface

```python
def workload(query, key, value, A_log):
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
        64
      ]
    },
    {
      "dtype": "bfloat16",
      "name": "key",
      "shape": [
        4,
        64,
        4096,
        64
      ]
    },
    {
      "dtype": "bfloat16",
      "name": "value",
      "shape": [
        4,
        64,
        4096,
        64
      ]
    },
    {
      "dtype": "float32",
      "name": "A_log",
      "shape": [
        4,
        64,
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
        64
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
    "value",
    "A_log"
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
          "id": "a",
          "kind": "Name"
        }
      ],
      "value": {
        "args": [
          {
            "args": [
              {
                "attr": "float32",
                "kind": "Attribute",
                "value": {
                  "id": "jnp",
                  "kind": "Name"
                }
              }
            ],
            "func": {
              "attr": "astype",
              "kind": "Attribute",
              "value": {
                "id": "A_log",
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
          "id": "log_a",
          "kind": "Name"
        }
      ],
      "value": {
        "args": [
          {
            "kind": "BinOp",
            "left": {
              "id": "a",
              "kind": "Name"
            },
            "op": {
              "kind": "Add"
            },
            "right": {
              "kind": null,
              "value": 1e-08
            }
          }
        ],
        "func": {
          "attr": "log",
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
          "id": "log_a_cumsum",
          "kind": "Name"
        }
      ],
      "value": {
        "args": [
          {
            "id": "log_a",
            "kind": "Name"
          }
        ],
        "func": {
          "attr": "cumsum",
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
      "kind": "Assign",
      "targets": [
        {
          "id": "diff",
          "kind": "Name"
        }
      ],
      "value": {
        "kind": "BinOp",
        "left": {
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
              }
            ],
            "kind": "Tuple"
          },
          "value": {
            "id": "log_a_cumsum",
            "kind": "Name"
          }
        },
        "op": {
          "kind": "Sub"
        },
        "right": {
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
            "id": "log_a_cumsum",
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
          "id": "L",
          "kind": "Name"
        }
      ],
      "value": {
        "args": [
          {
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
                "id": "diff",
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
        ],
        "func": {
          "attr": "exp",
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
          "id": "scores",
          "kind": "Name"
        }
      ],
      "value": {
        "args": [
          {
            "kind": null,
            "value": "bhsd,bhtd->bhst"
          },
          {
            "args": [
              {
                "attr": "float32",
                "kind": "Attribute",
                "value": {
                  "id": "jnp",
                  "kind": "Name"
                }
              }
            ],
            "func": {
              "attr": "astype",
              "kind": "Attribute",
              "value": {
                "id": "query",
                "kind": "Name"
              }
            },
            "keywords": [],
            "kind": "Call"
          },
          {
            "args": [
              {
                "attr": "float32",
                "kind": "Attribute",
                "value": {
                  "id": "jnp",
                  "kind": "Name"
                }
              }
            ],
            "func": {
              "attr": "astype",
              "kind": "Attribute",
              "value": {
                "id": "key",
                "kind": "Name"
              }
            },
            "keywords": [],
            "kind": "Call"
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
          "id": "scores",
          "kind": "Name"
        }
      ],
      "value": {
        "kind": "BinOp",
        "left": {
          "id": "scores",
          "kind": "Name"
        },
        "op": {
          "kind": "Mult"
        },
        "right": {
          "id": "L",
          "kind": "Name"
        }
      }
    },
    {
      "kind": "Assign",
      "targets": [
        {
          "id": "scores_sum",
          "kind": "Name"
        }
      ],
      "value": {
        "args": [
          {
            "id": "scores",
            "kind": "Name"
          }
        ],
        "func": {
          "attr": "sum",
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
      }
    },
    {
      "kind": "Assign",
      "targets": [
        {
          "id": "scores_sum",
          "kind": "Name"
        }
      ],
      "value": {
        "args": [
          {
            "comparators": [
              {
                "kind": null,
                "value": 1e-06
              }
            ],
            "kind": "Compare",
            "left": {
              "args": [
                {
                  "id": "scores_sum",
                  "kind": "Name"
                }
              ],
              "func": {
                "attr": "abs",
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
          },
          {
            "kind": null,
            "value": 1.0
          },
          {
            "id": "scores_sum",
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
          "id": "scores",
          "kind": "Name"
        }
      ],
      "value": {
        "kind": "BinOp",
        "left": {
          "id": "scores",
          "kind": "Name"
        },
        "op": {
          "kind": "Div"
        },
        "right": {
          "args": [
            {
              "args": [
                {
                  "id": "scores_sum",
                  "kind": "Name"
                }
              ],
              "func": {
                "attr": "abs",
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
              "kind": null,
              "value": 1.0
            }
          ],
          "func": {
            "attr": "maximum",
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
            "value": "bhst,bhtd->bhsd"
          },
          {
            "args": [
              {
                "attr": "dtype",
                "kind": "Attribute",
                "value": {
                  "id": "query",
                  "kind": "Name"
                }
              }
            ],
            "func": {
              "attr": "astype",
              "kind": "Attribute",
              "value": {
                "id": "scores",
                "kind": "Name"
              }
            },
            "keywords": [],
            "kind": "Call"
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

Mamba-2 SSD: structured linear attention with selective decay.

y = (L ⊙ (C B^T)) x
where L[i,j] = Π_{k=j+1}^{i} a_k for i > j, 1 for i=j, 0 for i<j
and a_k = exp(A_log_k) is the selective (input-dependent) decay.

The input generator, concrete test values, baseline implementation, optimized reference, correctness tests, and grading logic are hidden. Do not use `interpret=True` or a plain-JAX fallback.
