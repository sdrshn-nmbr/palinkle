# 13p_Cross_Entropy

Implement this pinned JAXBench workload as a normally lowered TPU Pallas kernel. Preserve the original workload semantics and original deployment shapes. The hidden verifier creates the exact inputs, computes the JAX baseline, checks normal Pallas lowering, captures a TPU profile, and compares performance against XLA.

Fused Linear + Cross-Entropy Loss — Llama 3.1 8B. From openxla/tokamax.

## Configuration

```json
{
  "batch_tokens": 8192,
  "hidden_dim": 4096,
  "model": "Llama-3.1-8B",
  "name": "llama3_8b_cross_entropy",
  "operator": "fused_cross_entropy",
  "vocab_size": 128256
}
```

## Required interface

```python
def workload(hidden, weight, labels):
    ...
```

## Tensor contract

```json
{
  "inputs": [
    {
      "dtype": "bfloat16",
      "name": "hidden",
      "shape": [
        8192,
        4096
      ]
    },
    {
      "dtype": "bfloat16",
      "name": "weight",
      "shape": [
        4096,
        128256
      ]
    },
    {
      "dtype": "int32",
      "name": "labels",
      "shape": [
        8192
      ]
    }
  ],
  "outputs": [
    {
      "dtype": "float32",
      "shape": []
    }
  ]
}
```

## Exact semantic contract

The following non-executable canonical AST defines operation order, constants, axes, layouts, padding, precision, and all other observable semantics. `Name` and `Attribute` nodes name mathematical/JAX operations; the hidden source implementation is not included.

```json
{
  "arguments": [
    "hidden",
    "weight",
    "labels"
  ],
  "body": [
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
            "id": "hidden",
            "kind": "Name"
          },
          {
            "id": "weight",
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
          "id": "log_probs",
          "kind": "Name"
        }
      ],
      "value": {
        "args": [
          {
            "id": "logits",
            "kind": "Name"
          }
        ],
        "func": {
          "attr": "log_softmax",
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
          "id": "one_hot",
          "kind": "Name"
        }
      ],
      "value": {
        "args": [
          {
            "id": "labels",
            "kind": "Name"
          },
          {
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
                "id": "logits",
                "kind": "Name"
              }
            }
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
          "id": "loss",
          "kind": "Name"
        }
      ],
      "value": {
        "kind": "UnaryOp",
        "op": {
          "kind": "USub"
        },
        "operand": {
          "args": [
            {
              "kind": "BinOp",
              "left": {
                "id": "one_hot",
                "kind": "Name"
              },
              "op": {
                "kind": "Mult"
              },
              "right": {
                "id": "log_probs",
                "kind": "Name"
              }
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
            }
          ],
          "kind": "Call"
        }
      }
    },
    {
      "kind": "Return",
      "value": {
        "args": [
          {
            "id": "loss",
            "kind": "Name"
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
        "keywords": [],
        "kind": "Call"
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

Fused linear projection + softmax cross-entropy loss.

The input generator, concrete test values, baseline implementation, optimized reference, correctness tests, and grading logic are hidden. Do not use `interpret=True` or a plain-JAX fallback.
