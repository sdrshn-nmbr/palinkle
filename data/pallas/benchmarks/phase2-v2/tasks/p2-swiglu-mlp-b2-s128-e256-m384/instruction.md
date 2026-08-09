Implement an authentic, normally lowered Pallas kernel in kernel.py.

The complete public contract is:
- Inputs: [{"dtype": "float32", "name": "x0", "range": [-1.0, 1.0], "shape": [2, 128, 256], "timing_range": [-1.0, 1.0]}, {"dtype": "float32", "name": "x1", "range": [-0.125, 0.125], "shape": [256, 384], "timing_range": [-0.125, 0.125]}, {"dtype": "float32", "name": "x2", "range": [-0.125, 0.125], "shape": [256, 384], "timing_range": [-0.125, 0.125]}, {"dtype": "float32", "name": "x3", "range": [-0.125, 0.125], "shape": [384, 256], "timing_range": [-0.125, 0.125]}]
- Equation: gate = silu(dot(x0, x1)); up = dot(x0, x2); output = dot(gate * up, x3)
- Output: shape [2, 128, 256], dtype float32
- Correctness tolerance: {"atol": 0.001, "rtol": 0.001}

Preserve the workload interface and run public checks. Do not use interpret mode or a plain-JAX fallback. The hidden verifier uses three fixed input seeds.
