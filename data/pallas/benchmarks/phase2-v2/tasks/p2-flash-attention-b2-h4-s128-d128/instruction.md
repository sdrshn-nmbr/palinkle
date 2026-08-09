Implement an authentic, normally lowered Pallas kernel in kernel.py.

The complete public contract is:
- Inputs: [{"dtype": "float32", "name": "x0", "range": [-1.0, 1.0], "shape": [2, 4, 128, 128], "timing_range": [-1.0, 1.0]}, {"dtype": "float32", "name": "x1", "range": [-1.0, 1.0], "shape": [2, 4, 128, 128], "timing_range": [-1.0, 1.0]}, {"dtype": "float32", "name": "x2", "range": [-1.0, 1.0], "shape": [2, 4, 128, 128], "timing_range": [-1.0, 1.0]}]
- Equation: scores[b,h,q,k] = dot(x0[b,h,q,:], x1[b,h,k,:]) / sqrt(head_dim); scores where k > q are -1e9; output = softmax(scores, axis -1) @ x2
- Output: shape [2, 4, 128, 128], dtype float32
- Correctness tolerance: {"atol": 0.001, "rtol": 0.001}

Preserve the workload interface and run public checks. Do not use interpret mode or a plain-JAX fallback. The hidden verifier uses three fixed input seeds.
