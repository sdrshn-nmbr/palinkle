# 12p_RMSNorm

Implement this pinned JAXBench workload as a normally lowered TPU Pallas kernel. Preserve the original workload semantics and original deployment shapes. The hidden verifier creates the exact inputs, computes the JAX baseline, checks normal Pallas lowering, captures a TPU profile, and compares performance against XLA.

RMSNorm — Llama-3.1-70B pre-attention normalization.

Root Mean Square Layer Normalization at Llama-3.1-70B scale.
Input shape: (batch=1, seq_len=2048, emb_dim=8192).
From MaxText layers/normalizations.py.

## Configuration

```json
{
  "batch": 8,
  "emb_dim": 8192,
  "epsilon": 1e-05,
  "model": "Llama-3.1-70B",
  "name": "llama3_70b_rmsnorm",
  "operator": "rms_norm",
  "seq_len": 4096
}
```

## Required interface

```python
def workload(x, scale):
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
        8,
        4096,
        8192
      ]
    },
    {
      "dtype": "bfloat16",
      "name": "scale",
      "shape": [
        8192
      ]
    }
  ],
  "outputs": [
    {
      "dtype": "bfloat16",
      "shape": [
        8,
        4096,
        8192
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
    "scale"
  ],
  "body": [
    {
      "kind": "Assign",
      "targets": [
        {
          "id": "x_f32",
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
            "attr": "float32",
            "kind": "Attribute",
            "value": {
              "id": "jnp",
              "kind": "Name"
            }
          }
        ],
        "func": {
          "attr": "asarray",
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
          "id": "mean2",
          "kind": "Name"
        }
      ],
      "value": {
        "args": [
          {
            "args": [
              {
                "id": "x_f32",
                "kind": "Name"
              }
            ],
            "func": {
              "attr": "square",
              "kind": "Attribute",
              "value": {
                "id": "lax",
                "kind": "Name"
              }
            },
            "keywords": [],
            "kind": "Call"
          }
        ],
        "func": {
          "attr": "mean",
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
          "id": "normed",
          "kind": "Name"
        }
      ],
      "value": {
        "kind": "BinOp",
        "left": {
          "id": "x_f32",
          "kind": "Name"
        },
        "op": {
          "kind": "Mult"
        },
        "right": {
          "args": [
            {
              "kind": "BinOp",
              "left": {
                "id": "mean2",
                "kind": "Name"
              },
              "op": {
                "kind": "Add"
              },
              "right": {
                "kind": "Subscript",
                "slice": {
                  "kind": null,
                  "value": "epsilon"
                },
                "value": {
                  "id": "CONFIG",
                  "kind": "Name"
                }
              }
            }
          ],
          "func": {
            "attr": "rsqrt",
            "kind": "Attribute",
            "value": {
              "id": "lax",
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
          "id": "normed",
          "kind": "Name"
        }
      ],
      "value": {
        "args": [
          {
            "id": "normed",
            "kind": "Name"
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
          "attr": "asarray",
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
        "kind": "BinOp",
        "left": {
          "id": "normed",
          "kind": "Name"
        },
        "op": {
          "kind": "Mult"
        },
        "right": {
          "id": "scale",
          "kind": "Name"
        }
      }
    }
  ],
  "format": "canonical_python_ast_semantics_v1",
  "helper_functions": {},
  "imports": {
    "jnp": "jax.numpy",
    "lax": "jax.lax"
  },
  "module_values": {},
  "unresolved_names": []
}
```

RMSNorm: x * rsqrt(mean(x^2) + eps) * scale

The input generator, concrete test values, baseline implementation, optimized reference, correctness tests, and grading logic are hidden. Do not use `interpret=True` or a plain-JAX fallback.
