"""Stable JAXBench prompt rendering and completion extraction."""

from __future__ import annotations

import ast
import hashlib
import re

SYSTEM_PALLAS_REQUIRED = (
    "You are an expert TPU Pallas kernel engineer. Return exactly one fenced "
    "Python module defining workload(*inputs). The measured implementation must "
    "reach jax.experimental.pallas.pallas_call. Do not include a plain-JAX "
    "fallback, tests, benchmark harness, or prose outside the code fence."
)

ANSWER_CONTRACT = (
    "The module must be syntactically valid, self-contained, and preserve the "
    "specified argument order, output structure, shapes, and dtypes."
)

_SAMPLE_JUNK = re.compile(
    r"<\|(?:end_message|content_model_end_sampling|eot_id|end_of_text)\|>"
)


def source_sha256(source: str) -> str:
    return hashlib.sha256(source.encode()).hexdigest()


def parses(source: str) -> bool:
    try:
        ast.parse(source)
    except SyntaxError:
        return False
    return True


def extract_code(completion: str) -> str | None:
    completion = _SAMPLE_JUNK.sub("", completion)
    candidates = re.findall(r"```(?:python|py)?\n(.*?)```", completion, re.DOTALL)
    tail = re.split(r"```(?:python|py)?\n", completion)
    if len(tail) > 1:
        candidates.append(tail[-1])
    cleaned = [
        _SAMPLE_JUNK.sub("", candidate).strip() + "\n"
        for candidate in candidates
        if candidate.strip()
    ]
    for require_parse in (True, False):
        for candidate in cleaned:
            if "def workload" not in candidate:
                continue
            if require_parse and not parses(candidate):
                continue
            return candidate
    return None


def spec_only(baseline_source: str) -> str:
    tree = ast.parse(baseline_source)
    parts: list[str] = []
    module_docstring = ast.get_docstring(tree)
    if module_docstring:
        parts.append(f'"""{module_docstring}"""')
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "CONFIG"
            for target in node.targets
        ):
            parts.append(ast.get_source_segment(baseline_source, node) or "")
        elif isinstance(node, ast.FunctionDef) and node.name == "workload":
            arguments = ast.unparse(node.args)
            docstring = ast.get_docstring(node)
            body = f'    """{docstring}"""\n' if docstring else ""
            parts.append(f"def workload({arguments}):\n{body}    ...")
    return "\n\n".join(part for part in parts if part) + "\n"


def render_prompt(
    *,
    workload: str,
    baseline_source: str,
    prompt_context: str,
) -> str:
    if prompt_context == "spec":
        context = spec_only(baseline_source)
        label = "I/O specification with the implementation withheld"
    elif prompt_context == "baseline":
        context = baseline_source
        label = "Reference implementation; diagnostic context only"
    else:
        raise ValueError(f"PROMPT_CONTEXT_INVALID: {prompt_context}")
    return (
        f"JAXBench workload: {workload}\n"
        f"{label}:\n```python\n{context}```\n\n{ANSWER_CONTRACT}"
    )
