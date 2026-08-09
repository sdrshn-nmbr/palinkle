Implement an authentic, normally lowered Pallas kernel in kernel.py.

The complete public contract is:
- Inputs: [{"dtype": "float32", "name": "x0", "range": [-1.0, 1.0], "shape": [384, 256], "timing_range": [-1.0, 1.0]}, {"dtype": "float32", "name": "x1", "range": [-0.25, 0.25], "shape": [256, 256], "timing_range": [-0.25, 0.25]}, {"dtype": "float32", "name": "x2", "range": [-0.25, 0.25], "shape": [256], "timing_range": [-0.25, 0.25]}]
- Equation: values = matmul(x0, x1) + x2; first = values * tanh(logaddexp(values, 0)); output = first * tanh(logaddexp(first, 0))
- Output: shape [384, 256], dtype float32
- Correctness tolerance: {"atol": 0.001, "rtol": 0.001}

Preserve the workload interface and run public checks. Do not use interpret mode or a plain-JAX fallback. The hidden verifier uses three fixed input seeds.
