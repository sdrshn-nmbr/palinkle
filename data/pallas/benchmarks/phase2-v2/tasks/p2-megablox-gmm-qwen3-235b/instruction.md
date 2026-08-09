Implement an authentic, normally lowered Pallas kernel in kernel.py.

The complete public contract is:
- Inputs: [{"dtype": "bfloat16", "name": "x0", "range": [-0.015625, 0.015625], "shape": [32768, 4096], "timing_range": [-1e-08, 1e-08]}, {"dtype": "bfloat16", "name": "x1", "range": [-0.015625, 0.015625], "shape": [128, 4096, 1536], "timing_range": [-1e-08, 1e-08]}, {"dtype": "int32", "name": "x2", "range": [256, 257], "shape": [128], "timing_range": [256, 257]}]
- Equation: x2 contains contiguous group sizes summing to rows(x0); each output group slice is matmul(the matching x0 rows, x1[group])
- Output: shape [32768, 1536], dtype bfloat16
- Correctness tolerance: {"atol": 0.001, "rtol": 0.001}

Preserve the workload interface and run public checks. Do not use interpret mode or a plain-JAX fallback. The hidden verifier uses three fixed input seeds.
