# 21k_Gemm_Divide_Sum_Scaling

Implement this pinned JAXBench workload as a normally lowered TPU Pallas kernel. Preserve the original workload semantics and original deployment shapes. The hidden verifier creates the exact inputs, computes the JAX baseline, checks normal Pallas lowering, captures a TPU profile, and compares performance against XLA.

14_Gemm_Divide_Sum_Scaling — JAXBench fused operator workload.

## Configuration

```json
{
  "batch_size": 4096,
  "hidden_size": 8192,
  "input_size": 8192,
  "name": "14_Gemm_Divide_Sum_Scaling",
  "scaling_factor": 1.5
}
```

## Required interface

```python
def workload(x, weight):
    ...
```

## Tensor contract

```json
{
  "inputs": [
    {
      "dtype": "float32",
      "name": "x",
      "shape": [
        4096,
        8192
      ]
    },
    {
      "dtype": "float32",
      "name": "weight",
      "shape": [
        8192,
        8192
      ]
    }
  ],
  "outputs": [
    {
      "dtype": "float32",
      "shape": [
        4096,
        1
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
    "weight"
  ],
  "body": [
    {
      "kind": "Assign",
      "targets": [
        {
          "id": "x",
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
            "attr": "T",
            "kind": "Attribute",
            "value": {
              "id": "weight",
              "kind": "Name"
            }
          }
        ],
        "func": {
          "attr": "dot_general",
          "kind": "Attribute",
          "value": {
            "id": "lax",
            "kind": "Name"
          }
        },
        "keywords": [
          {
            "arg": "dimension_numbers",
            "kind": "keyword",
            "value": {
              "elts": [
                {
                  "elts": [
                    {
                      "elts": [
                        {
                          "kind": null,
                          "value": 1
                        }
                      ],
                      "kind": "Tuple"
                    },
                    {
                      "elts": [
                        {
                          "kind": null,
                          "value": 0
                        }
                      ],
                      "kind": "Tuple"
                    }
                  ],
                  "kind": "Tuple"
                },
                {
                  "elts": [
                    {
                      "elts": [],
                      "kind": "Tuple"
                    },
                    {
                      "elts": [],
                      "kind": "Tuple"
                    }
                  ],
                  "kind": "Tuple"
                }
              ],
              "kind": "Tuple"
            }
          },
          {
            "arg": "precision",
            "kind": "keyword",
            "value": {
              "attr": "HIGHEST",
              "kind": "Attribute",
              "value": {
                "attr": "Precision",
                "kind": "Attribute",
                "value": {
                  "id": "lax",
                  "kind": "Name"
                }
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
          "id": "x",
          "kind": "Name"
        }
      ],
      "value": {
        "kind": "BinOp",
        "left": {
          "id": "x",
          "kind": "Name"
        },
        "op": {
          "kind": "Div"
        },
        "right": {
          "kind": null,
          "value": 2.0
        }
      }
    },
    {
      "kind": "Assign",
      "targets": [
        {
          "id": "x",
          "kind": "Name"
        }
      ],
      "value": {
        "args": [
          {
            "id": "x",
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
              "kind": null,
              "value": 1
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
          "id": "x",
          "kind": "Name"
        }
      ],
      "value": {
        "kind": "BinOp",
        "left": {
          "id": "x",
          "kind": "Name"
        },
        "op": {
          "kind": "Mult"
        },
        "right": {
          "kind": null,
          "value": 1.5
        }
      }
    },
    {
      "kind": "Return",
      "value": {
        "id": "x",
        "kind": "Name"
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

Gemm + Divide + Sum + Scaling.

The input generator, concrete test values, baseline implementation, optimized reference, correctness tests, and grading logic are hidden. Do not use `interpret=True` or a plain-JAX fallback.
