Implement an authentic, normally lowered Pallas kernel in kernel.py.

The complete public contract is:
- Inputs: [{"dtype": "float32", "name": "x0", "range": [-0.5, 0.5], "shape": [512, 256], "timing_range": [-0.5, 0.5]}, {"dtype": "float32", "name": "x1", "range": [-0.25, 0.25], "shape": [256, 384], "timing_range": [-0.25, 0.25]}, {"dtype": "float32", "name": "x2", "range": [-0.5, 0.5], "shape": [384], "timing_range": [-0.5, 0.5]}]
- Equation: output = maximum(matmul(x0, x1) + x2, 0)
- Output: shape [512, 384], dtype float32
- Correctness tolerance: {"atol": 0.001, "rtol": 0.001}

Preserve the workload interface and run public checks. Do not use interpret mode or a plain-JAX fallback. The hidden verifier uses three fixed input seeds.
