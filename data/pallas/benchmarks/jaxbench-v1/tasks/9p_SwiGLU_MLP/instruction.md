# 9p_SwiGLU_MLP

Implement this pinned JAXBench workload as a normally lowered TPU Pallas kernel. Preserve the original workload semantics and original deployment shapes. The hidden verifier creates the exact inputs, computes the JAX baseline, checks normal Pallas lowering, captures a TPU profile, and compares performance against XLA.

SwiGLU MLP — Llama 3.1 70B. Extracted from MaxText.

## Configuration

```json
{
  "batch": 2,
  "emb_dim": 8192,
  "mlp_dim": 28672,
  "model": "Llama-3.1-70B",
  "name": "llama3_70b_swiglu",
  "operator": "swiglu_mlp",
  "seq_len": 4096
}
```

## Required interface

```python
def workload(x, gate_kernel, up_kernel, down_kernel):
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
        8192
      ]
    },
    {
      "dtype": "bfloat16",
      "name": "gate_kernel",
      "shape": [
        8192,
        28672
      ]
    },
    {
      "dtype": "bfloat16",
      "name": "up_kernel",
      "shape": [
        8192,
        28672
      ]
    },
    {
      "dtype": "bfloat16",
      "name": "down_kernel",
      "shape": [
        28672,
        8192
      ]
    }
  ],
  "outputs": [
    {
      "dtype": "bfloat16",
      "shape": [
        2,
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
    "gate_kernel",
    "up_kernel",
    "down_kernel"
  ],
  "body": [
    {
      "kind": "Assign",
      "targets": [
        {
          "id": "gate",
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
                "id": "gate_kernel",
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
          "id": "up",
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
            "id": "up_kernel",
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
      "kind": "Return",
      "value": {
        "args": [
          {
            "kind": "BinOp",
            "left": {
              "id": "gate",
              "kind": "Name"
            },
            "op": {
              "kind": "Mult"
            },
            "right": {
              "id": "up",
              "kind": "Name"
            }
          },
          {
            "id": "down_kernel",
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
  "helper_functions": {},
  "imports": {
    "jax": "jax",
    "jnp": "jax.numpy"
  },
  "module_values": {},
  "unresolved_names": []
}
```

SwiGLU: output = (SiLU(x @ gate) * (x @ up)) @ down

The input generator, concrete test values, baseline implementation, optimized reference, correctness tests, and grading logic are hidden. Do not use `interpret=True` or a plain-JAX fallback.
