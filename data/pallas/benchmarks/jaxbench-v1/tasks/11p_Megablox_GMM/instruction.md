# 11p_Megablox_GMM

Implement this pinned JAXBench workload as a normally lowered TPU Pallas kernel. Preserve the original workload semantics and original deployment shapes. The hidden verifier creates the exact inputs, computes the JAX baseline, checks normal Pallas lowering, captures a TPU profile, and compares performance against XLA.

Grouped Matrix Multiply (Megablox GMM) — Qwen3-235B-A22B MoE dimensions.

Reference grouped matmul: for each expert group, slice the input tokens
and multiply with that expert's weight matrix. Core primitive for MoE layers.
From JAX experimental pallas ops (reference_gmm).

## Configuration

```json
{
  "emb_dim": 4096,
  "model": "Qwen3-235B-A22B",
  "moe_mlp_dim": 1536,
  "name": "megablox_gmm_qwen3_235b",
  "num_experts": 128,
  "num_experts_per_tok": 8,
  "operator": "grouped_matmul",
  "seq_len": 4096
}
```

## Required interface

```python
def workload(lhs, rhs, group_sizes, max_expert_size):
    ...
```

## Tensor contract

```json
{
  "inputs": [
    {
      "dtype": "bfloat16",
      "name": "lhs",
      "shape": [
        32768,
        4096
      ]
    },
    {
      "dtype": "bfloat16",
      "name": "rhs",
      "shape": [
        128,
        4096,
        1536
      ]
    },
    {
      "dtype": "int32",
      "name": "group_sizes",
      "shape": [
        128
      ]
    },
    {
      "dtype": "int32",
      "name": "max_expert_size",
      "shape": []
    }
  ],
  "outputs": [
    {
      "dtype": "bfloat16",
      "shape": [
        32768,
        1536
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
    "lhs",
    "rhs",
    "group_sizes",
    "max_expert_size"
  ],
  "body": [
    {
      "kind": "Assign",
      "targets": [
        {
          "id": "G",
          "kind": "Name"
        }
      ],
      "value": {
        "kind": "Subscript",
        "slice": {
          "kind": null,
          "value": 0
        },
        "value": {
          "attr": "shape",
          "kind": "Attribute",
          "value": {
            "id": "rhs",
            "kind": "Name"
          }
        }
      }
    },
    {
      "kind": "Assign",
      "targets": [
        {
          "elts": [
            {
              "id": "M",
              "kind": "Name"
            },
            {
              "id": "K",
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
          "id": "lhs",
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
          "kind": null,
          "value": 2
        },
        "value": {
          "attr": "shape",
          "kind": "Attribute",
          "value": {
            "id": "rhs",
            "kind": "Name"
          }
        }
      }
    },
    {
      "kind": "Assign",
      "targets": [
        {
          "id": "group_ends",
          "kind": "Name"
        }
      ],
      "value": {
        "args": [
          {
            "id": "group_sizes",
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
        "keywords": [],
        "kind": "Call"
      }
    },
    {
      "kind": "Assign",
      "targets": [
        {
          "id": "group_starts",
          "kind": "Name"
        }
      ],
      "value": {
        "args": [
          {
            "elts": [
              {
                "args": [
                  {
                    "kind": null,
                    "value": 1
                  }
                ],
                "func": {
                  "attr": "zeros",
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
                      "attr": "int32",
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
              {
                "kind": "Subscript",
                "slice": {
                  "kind": "Slice",
                  "lower": null,
                  "step": null,
                  "upper": {
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
                "value": {
                  "id": "group_ends",
                  "kind": "Name"
                }
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
        "keywords": [],
        "kind": "Call"
      }
    },
    {
      "kind": "Assign",
      "targets": [
        {
          "id": "res_flat",
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
                  "id": "M",
                  "kind": "Name"
                },
                "op": {
                  "kind": "Add"
                },
                "right": {
                  "id": "max_expert_size",
                  "kind": "Name"
                }
              },
              {
                "id": "N",
                "kind": "Name"
              }
            ],
            "kind": "Tuple"
          }
        ],
        "func": {
          "attr": "zeros",
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
              "attr": "dtype",
              "kind": "Attribute",
              "value": {
                "id": "lhs",
                "kind": "Name"
              }
            }
          }
        ],
        "kind": "Call"
      }
    },
    {
      "args": {
        "args": [
          {
            "annotation": null,
            "arg": "carry_res_flat",
            "kind": "arg"
          },
          {
            "annotation": null,
            "arg": "i",
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
              "id": "start",
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
              "id": "group_starts",
              "kind": "Name"
            }
          }
        },
        {
          "kind": "Assign",
          "targets": [
            {
              "id": "count",
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
              "id": "group_sizes",
              "kind": "Name"
            }
          }
        },
        {
          "kind": "Assign",
          "targets": [
            {
              "id": "expert_lhs",
              "kind": "Name"
            }
          ],
          "value": {
            "args": [
              {
                "id": "lhs",
                "kind": "Name"
              },
              {
                "elts": [
                  {
                    "id": "start",
                    "kind": "Name"
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
                    "id": "max_expert_size",
                    "kind": "Name"
                  },
                  {
                    "id": "K",
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
              "id": "expert_rhs",
              "kind": "Name"
            }
          ],
          "value": {
            "kind": "Subscript",
            "slice": {
              "elts": [
                {
                  "id": "i",
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
                  "lower": null,
                  "step": null,
                  "upper": null
                }
              ],
              "kind": "Tuple"
            },
            "value": {
              "id": "rhs",
              "kind": "Name"
            }
          }
        },
        {
          "kind": "Assign",
          "targets": [
            {
              "id": "res",
              "kind": "Name"
            }
          ],
          "value": {
            "args": [
              {
                "id": "expert_lhs",
                "kind": "Name"
              },
              {
                "id": "expert_rhs",
                "kind": "Name"
              }
            ],
            "func": {
              "attr": "dot",
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
                "id": "count",
                "kind": "Name"
              }
            ],
            "kind": "Compare",
            "left": {
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
                  "elts": [
                    {
                      "id": "max_expert_size",
                      "kind": "Name"
                    },
                    {
                      "id": "N",
                      "kind": "Name"
                    }
                  ],
                  "kind": "Tuple"
                },
                {
                  "kind": null,
                  "value": 0
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
              "id": "res_masked",
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
                "id": "res",
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
          "kind": "Assign",
          "targets": [
            {
              "id": "current_slice",
              "kind": "Name"
            }
          ],
          "value": {
            "args": [
              {
                "id": "carry_res_flat",
                "kind": "Name"
              },
              {
                "elts": [
                  {
                    "id": "start",
                    "kind": "Name"
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
                    "id": "max_expert_size",
                    "kind": "Name"
                  },
                  {
                    "id": "N",
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
              "id": "updated_slice",
              "kind": "Name"
            }
          ],
          "value": {
            "kind": "BinOp",
            "left": {
              "id": "current_slice",
              "kind": "Name"
            },
            "op": {
              "kind": "Add"
            },
            "right": {
              "args": [
                {
                  "attr": "dtype",
                  "kind": "Attribute",
                  "value": {
                    "id": "carry_res_flat",
                    "kind": "Name"
                  }
                }
              ],
              "func": {
                "attr": "astype",
                "kind": "Attribute",
                "value": {
                  "id": "res_masked",
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
              "id": "carry_res_flat",
              "kind": "Name"
            }
          ],
          "value": {
            "args": [
              {
                "id": "carry_res_flat",
                "kind": "Name"
              },
              {
                "id": "updated_slice",
                "kind": "Name"
              },
              {
                "elts": [
                  {
                    "id": "start",
                    "kind": "Name"
                  },
                  {
                    "kind": null,
                    "value": 0
                  }
                ],
                "kind": "Tuple"
              }
            ],
            "func": {
              "attr": "dynamic_update_slice",
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
          "kind": "Return",
          "value": {
            "elts": [
              {
                "id": "carry_res_flat",
                "kind": "Name"
              },
              {
                "kind": null,
                "value": null
              }
            ],
            "kind": "Tuple"
          }
        }
      ],
      "decorator_list": [],
      "kind": "FunctionDef",
      "name": "body_fun",
      "returns": null,
      "type_params": []
    },
    {
      "kind": "Assign",
      "targets": [
        {
          "elts": [
            {
              "id": "res_flat",
              "kind": "Name"
            },
            {
              "id": "_",
              "kind": "Name"
            }
          ],
          "kind": "Tuple"
        }
      ],
      "value": {
        "args": [
          {
            "id": "body_fun",
            "kind": "Name"
          },
          {
            "id": "res_flat",
            "kind": "Name"
          },
          {
            "args": [
              {
                "id": "G",
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
          "attr": "scan",
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
      "kind": "Return",
      "value": {
        "kind": "Subscript",
        "slice": {
          "elts": [
            {
              "kind": "Slice",
              "lower": null,
              "step": null,
              "upper": {
                "id": "M",
                "kind": "Name"
              }
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
          "id": "res_flat",
          "kind": "Name"
        }
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

Jittable grouped matmul using static shapes and masking.

Computes dot product for each group with static slice sizes to allow JIT.

The input generator, concrete test values, baseline implementation, optimized reference, correctness tests, and grading logic are hidden. Do not use `interpret=True` or a plain-JAX fallback.
