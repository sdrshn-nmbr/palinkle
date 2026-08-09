# Pallas task API

Edit `kernel.py`. It must define one complete `workload(*inputs)` implementation.
Use `pl.BlockSpec(block_shape, index_map)` in that order. The kernel must use a
reachable `pl.pallas_call`, must not use `interpret=True`, and must not include a
plain-JAX fallback.

Candidate code runs under a restricted, deterministic Python subset. Imports,
top-level statements, calls, bindings, and mutation targets are checked by the
public `candidate_policy.py`; it is byte-identical to the authoritative verifier
policy. Run `python dev_check.py kernel.py` for the exhaustive safe-language check
and the public Pallas API checks. TPU correctness and performance tests are hidden.
