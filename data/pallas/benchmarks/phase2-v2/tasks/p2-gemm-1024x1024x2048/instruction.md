Implement an authentic, normally lowered Pallas kernel in kernel.py.

The complete public contract is:
- Inputs: [{"dtype": "bfloat16", "name": "x0", "range": [-0.5, 0.5], "shape": [1024, 1024], "timing_range": [-0.5, 0.5]}, {"dtype": "bfloat16", "name": "x1", "range": [-0.125, 0.125], "shape": [1024, 2048], "timing_range": [-0.125, 0.125]}]
- Equation: output = matmul(x0, x1) with float32 accumulation
- Output: shape [1024, 2048], dtype bfloat16
- Correctness tolerance: {"atol": 0.001, "rtol": 0.001}

Preserve the workload interface and run public checks. Do not use interpret mode or a plain-JAX fallback. The hidden verifier uses three fixed input seeds.
