# 15p_RetNet_Retention

Implement this pinned JAXBench workload as a normally lowered TPU Pallas kernel. Preserve the original workload semantics and original deployment shapes. The hidden verifier creates the exact inputs, computes the JAX baseline, checks normal Pallas lowering, captures a TPU profile, and compares performance against XLA.

Multi-Scale Retention — Microsoft RetNet.

Replaces softmax attention with retention: a causal linear attention mechanism
with fixed exponential decay per head. Different heads use different decay rates
(multi-scale), giving each head a different "memory horizon".

Paper: "Retentive Network: A Successor to Transformer" (Sun et al., 2023)
Used in RetNet models and influenced Mamba-2, GLA, and other recent architectures.

Config based on RetNet-6.7B from the paper.

## Configuration

```json
{
  "batch": 4,
  "d_model": 4096,
  "head_dim": 256,
  "model": "RetNet-6.7B",
  "name": "retnet_6_7b_retention",
  "num_heads": 16,
  "operator": "multi_scale_retention",
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
        16,
        4096,
        256
      ]
    },
    {
      "dtype": "bfloat16",
      "name": "key",
      "shape": [
        4,
        16,
        4096,
        256
      ]
    },
    {
      "dtype": "bfloat16",
      "name": "value",
      "shape": [
        4,
        16,
        4096,
        256
      ]
    }
  ],
  "outputs": [
    {
      "dtype": "bfloat16",
      "shape": [
        4,
        16,
        4096,
        256
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
          "id": "gammas",
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
          "kind": "Sub"
        },
        "right": {
          "args": [
            {
              "kind": "BinOp",
              "left": {
                "kind": "UnaryOp",
                "op": {
                  "kind": "USub"
                },
                "operand": {
                  "kind": null,
                  "value": 5.0
                }
              },
              "op": {
                "kind": "Sub"
              },
              "right": {
                "args": [
                  {
                    "id": "H",
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
            }
          ],
          "func": {
            "attr": "exp2",
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
          "id": "positions",
          "kind": "Name"
        }
      ],
      "value": {
        "args": [
          {
            "id": "S",
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
          "id": "distance",
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
                "kind": null,
                "value": null
              }
            ],
            "kind": "Tuple"
          },
          "value": {
            "id": "positions",
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
            "id": "positions",
            "kind": "Name"
          }
        }
      }
    },
    {
      "kind": "Assign",
      "targets": [
        {
          "id": "causal_mask",
          "kind": "Name"
        }
      ],
      "value": {
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
            "comparators": [
              {
                "kind": null,
                "value": 0
              }
            ],
            "kind": "Compare",
            "left": {
              "id": "distance",
              "kind": "Name"
            },
            "ops": [
              {
                "kind": "GtE"
              }
            ]
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
          "id": "log_gamma",
          "kind": "Name"
        }
      ],
      "value": {
        "args": [
          {
            "id": "gammas",
            "kind": "Name"
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
          "id": "decay",
          "kind": "Name"
        }
      ],
      "value": {
        "args": [
          {
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
                    "kind": null,
                    "value": null
                  },
                  {
                    "kind": null,
                    "value": null
                  }
                ],
                "kind": "Tuple"
              },
              "value": {
                "id": "log_gamma",
                "kind": "Name"
              }
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
                "args": [
                  {
                    "id": "distance",
                    "kind": "Name"
                  },
                  {
                    "kind": null,
                    "value": 0.0
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
          "id": "decay",
          "kind": "Name"
        }
      ],
      "value": {
        "kind": "BinOp",
        "left": {
          "id": "decay",
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
            "id": "causal_mask",
            "kind": "Name"
          }
        }
      }
    },
    {
      "kind": "Assign",
      "targets": [
        {
          "id": "qk",
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
          "id": "qk",
          "kind": "Name"
        }
      ],
      "value": {
        "kind": "BinOp",
        "left": {
          "id": "qk",
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
            "id": "decay",
            "kind": "Name"
          }
        }
      }
    },
    {
      "kind": "Assign",
      "targets": [
        {
          "id": "retention_sum",
          "kind": "Name"
        }
      ],
      "value": {
        "args": [
          {
            "args": [
              {
                "id": "qk",
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
          "id": "retention_sum",
          "kind": "Name"
        }
      ],
      "value": {
        "args": [
          {
            "id": "retention_sum",
            "kind": "Name"
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
    },
    {
      "kind": "Assign",
      "targets": [
        {
          "id": "qk",
          "kind": "Name"
        }
      ],
      "value": {
        "kind": "BinOp",
        "left": {
          "id": "qk",
          "kind": "Name"
        },
        "op": {
          "kind": "Div"
        },
        "right": {
          "id": "retention_sum",
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
                "id": "qk",
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
    "jnp": "jax.numpy"
  },
  "module_values": {},
  "unresolved_names": []
}
```

Multi-scale retention with per-head exponential decay.

Retention(X) = (Q K^T ⊙ D) V
where D[i,j] = γ^(i-j) if i >= j, else 0

Each head has a different decay rate γ_h, creating a multi-scale
representation: some heads attend locally, others globally.

The input generator, concrete test values, baseline implementation, optimized reference, correctness tests, and grading logic are hidden. Do not use `interpret=True` or a plain-JAX fallback.
