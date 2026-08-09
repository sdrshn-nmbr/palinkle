Implement an authentic, normally lowered Pallas kernel in kernel.py.

The complete public contract is:
- Inputs: [{"dtype": "float32", "name": "x0", "range": [-2.0, 2.0], "shape": [4, 128, 512], "timing_range": [-2.0, 2.0]}, {"dtype": "float32", "name": "x1", "range": [0.5, 1.5], "shape": [512], "timing_range": [0.5, 1.5]}]
- Equation: x_f32 = float32(x0); output = cast(x_f32 * rsqrt(mean(square(x_f32), axis -1, keepdims true) + 1e-5), dtype(x0)) * x1
- Output: shape [4, 128, 512], dtype float32
- Correctness tolerance: {"atol": 0.001, "rtol": 0.001}

Preserve the workload interface and run public checks. Do not use interpret mode or a plain-JAX fallback. The hidden verifier uses three fixed input seeds.
