"""Public, pure static policy for Phase 2 candidate modules."""

from __future__ import annotations

import ast


_ALLOWED_IMPORTS = {
    ("import", "jax", None),
    ("import", "jax.numpy", "jnp"),
    ("from", "jax.experimental", "pallas", "pl"),
    ("from", "jax.experimental.pallas", "tpu", "pltpu"),
    (
        "from",
        "jax.experimental.pallas.ops.tpu.megablox",
        "gmm",
        None,
    ),
}
_ALLOWED_BARE_CALLS = {"float", "gmm", "int", "len", "range", "tuple"}
_RESERVED_BINDINGS = {
    "jax",
    "jnp",
    "pl",
    "pltpu",
    *_ALLOWED_BARE_CALLS,
}
_ALLOWED_JNP_CALLS = {
    "jnp.abs",
    "jnp.add",
    "jnp.arange",
    "jnp.array",
    "jnp.asarray",
    "jnp.clip",
    "jnp.concatenate",
    "jnp.cos",
    "jnp.divide",
    "jnp.dot",
    "jnp.einsum",
    "jnp.exp",
    "jnp.exp2",
    "jnp.expm1",
    "jnp.full",
    "jnp.full_like",
    "jnp.log",
    "jnp.log1p",
    "jnp.log2",
    "jnp.logaddexp",
    "jnp.matmul",
    "jnp.max",
    "jnp.maximum",
    "jnp.mean",
    "jnp.min",
    "jnp.minimum",
    "jnp.multiply",
    "jnp.ones",
    "jnp.ones_like",
    "jnp.power",
    "jnp.reshape",
    "jnp.sign",
    "jnp.sin",
    "jnp.sqrt",
    "jnp.square",
    "jnp.stack",
    "jnp.subtract",
    "jnp.sum",
    "jnp.swapaxes",
    "jnp.take",
    "jnp.take_along_axis",
    "jnp.tanh",
    "jnp.transpose",
    "jnp.where",
    "jnp.zeros",
    "jnp.zeros_like",
}
_ALLOWED_PL_CALLS = {
    "pl.BlockSpec",
    "pl.pallas_call",
    "pl.program_id",
    "pl.when",
}
_ALLOWED_PLTPU_CALLS = {
    "pltpu.CompilerParams",
    "pltpu.PrefetchScalarGridSpec",
    "pltpu.VMEM",
}
_FORBIDDEN_CAPABILITY_ATTRIBUTES = {
    "enable_debug_checks",
    "force_tpu_interpret_mode",
    "fromfile",
    "load",
    "lower_as_mlir",
    "pallas_export_experimental",
    "reset_tpu_interpret_mode_state",
    "save",
    "savez",
    "savez_compressed",
    "set_tpu_interpret_mode",
    "tofile",
}
_FORBIDDEN_NODES = (
    ast.AsyncFunctionDef,
    ast.AsyncFor,
    ast.AsyncWith,
    ast.Await,
    ast.ClassDef,
    ast.Delete,
    ast.Global,
    ast.Match,
    ast.Nonlocal,
    ast.Raise,
    ast.Try,
    ast.TryStar,
    ast.TypeAlias,
    ast.With,
    ast.Yield,
    ast.YieldFrom,
)


def _call_path(node: ast.Call) -> str:
    value: ast.AST = node.func
    parts: list[str] = []
    while isinstance(value, ast.Attribute):
        parts.append(value.attr)
        value = value.value
    if isinstance(value, ast.Name):
        parts.append(value.id)
    return ".".join(reversed(parts))


def _import_key(node: ast.Import | ast.ImportFrom) -> tuple[str, ...] | None:
    if isinstance(node, ast.Import):
        if len(node.names) != 1:
            return None
        alias = node.names[0]
        return ("import", alias.name, alias.asname)
    if node.level or node.module is None or len(node.names) != 1:
        return None
    alias = node.names[0]
    return ("from", node.module, alias.name, alias.asname)


def _function_signature_is_static(node: ast.FunctionDef) -> bool:
    arguments = [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]
    return not (
        node.decorator_list
        or node.type_params
        or node.returns is not None
        or node.type_comment is not None
        or node.args.defaults
        or any(default is not None for default in node.args.kw_defaults)
        or any(argument.annotation is not None for argument in arguments)
        or (node.args.vararg is not None and node.args.vararg.annotation is not None)
        or (node.args.kwarg is not None and node.args.kwarg.annotation is not None)
    )


def _flatten_assignment_target(node: ast.AST) -> list[ast.AST]:
    if isinstance(node, (ast.Tuple, ast.List)):
        return [
            item
            for element in node.elts
            for item in _flatten_assignment_target(element)
        ]
    if isinstance(node, ast.Starred):
        return _flatten_assignment_target(node.value)
    return [node]


def _constant_targets(node: ast.Assign) -> list[str] | None:
    names: list[str] = []
    for target in node.targets:
        for value in _flatten_assignment_target(target):
            if not isinstance(value, ast.Name) or not value.id.isupper():
                return None
            names.append(value.id)
    try:
        ast.literal_eval(node.value)
    except (ValueError, TypeError):
        return None
    return names


def _binding_target_error(
    targets: list[ast.AST], *, allow_ref_subscript: bool
) -> str | None:
    flattened = [
        item for target in targets for item in _flatten_assignment_target(target)
    ]
    if any(isinstance(target, ast.Attribute) for target in flattened):
        return "CANDIDATE_ATTRIBUTE_MUTATION_NOT_ALLOWED"
    if any(
        isinstance(target, ast.Name) and target.id in _RESERVED_BINDINGS
        for target in flattened
    ):
        return "CANDIDATE_RESERVED_NAME_REBIND_NOT_ALLOWED"
    if any(
        isinstance(target, ast.Name) and target.id.endswith("_ref")
        for target in flattened
    ):
        return "CANDIDATE_REFERENCE_REBIND_NOT_ALLOWED"
    for target in flattened:
        if not isinstance(target, ast.Subscript):
            continue
        if not (
            allow_ref_subscript
            and isinstance(target.value, ast.Name)
            and target.value.id.endswith("_ref")
        ):
            return "CANDIDATE_SUBSCRIPT_MUTATION_NOT_ALLOWED"
    if any(
        not isinstance(target, (ast.Name, ast.Subscript)) for target in flattened
    ):
        return "CANDIDATE_MUTATION_TARGET_NOT_ALLOWED"
    return None


def _call_is_numeric_or_pallas(
    node: ast.Call, *, allowed_entrypoints: tuple[str, ...]
) -> bool:
    path = _call_path(node)
    allowed_exact = {
        "jax.ShapeDtypeStruct",
        "pl.BlockSpec",
        "pl.pallas_call",
        *allowed_entrypoints,
        *(entrypoint.rsplit(".", 1)[-1] for entrypoint in allowed_entrypoints),
    }
    if path in allowed_exact or path in _ALLOWED_BARE_CALLS:
        return True
    if path in _ALLOWED_JNP_CALLS:
        return True
    if path.startswith(("jax.lax.", "jax.nn.", "jax.scipy.")):
        return True
    if path in _ALLOWED_PL_CALLS or path in _ALLOWED_PLTPU_CALLS:
        return True
    if isinstance(node.func, ast.Attribute) and node.func.attr == "astype":
        return True
    return (
        not path
        and isinstance(node.func, ast.Call)
        and _call_path(node.func) == "pl.pallas_call"
    )


def _names_loaded(node: ast.AST) -> set[str]:
    return {
        child.id
        for child in ast.walk(node)
        if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load)
    }


def _workload_normal_form_error(
    tree: ast.Module, *, allowed_entrypoints: tuple[str, ...]
) -> str | None:
    workloads = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "workload"
    ]
    if len(workloads) != 1:
        return "CANDIDATE_WORKLOAD_REQUIRED"
    workload = workloads[0]
    parameters = [
        argument.arg
        for argument in [*workload.args.posonlyargs, *workload.args.args]
    ]
    body = list(workload.body)
    if body and isinstance(body[0], ast.Expr) and isinstance(
        body[0].value, ast.Constant
    ):
        body = body[1:]
    if not body or not isinstance(body[-1], ast.Return) or body[-1].value is None:
        return "CANDIDATE_WORKLOAD_DIRECT_RETURN_REQUIRED"
    for statement in body[:-1]:
        if not isinstance(statement, ast.Assign):
            return "CANDIDATE_WORKLOAD_STATIC_SETUP_ONLY"
        if _names_loaded(statement.value).intersection(parameters):
            return "CANDIDATE_WORKLOAD_ARRAY_SETUP_NOT_ALLOWED"
    returned = body[-1].value
    if not isinstance(returned, ast.Call):
        return "CANDIDATE_WORKLOAD_DIRECT_RETURN_REQUIRED"
    if isinstance(returned.func, ast.Call):
        if _call_path(returned.func) != "pl.pallas_call":
            return "CANDIDATE_WORKLOAD_DIRECT_PALLAS_RETURN_REQUIRED"
        if returned.keywords or [
            argument.id if isinstance(argument, ast.Name) else None
            for argument in returned.args
        ] != parameters:
            return "CANDIDATE_WORKLOAD_ORIGINAL_INPUTS_REQUIRED"
        return None
    entrypoints = {
        *allowed_entrypoints,
        *(entrypoint.rsplit(".", 1)[-1] for entrypoint in allowed_entrypoints),
        "gmm",
    }
    if _call_path(returned) not in entrypoints:
        return "CANDIDATE_WORKLOAD_DIRECT_PALLAS_RETURN_REQUIRED"
    if len(returned.args) < len(parameters) or [
        argument.id if isinstance(argument, ast.Name) else None
        for argument in returned.args[: len(parameters)]
    ] != parameters:
        return "CANDIDATE_WORKLOAD_ORIGINAL_INPUTS_REQUIRED"
    for value in [*returned.args[len(parameters) :], *(item.value for item in returned.keywords)]:
        if _names_loaded(value).intersection(parameters):
            return "CANDIDATE_WORKLOAD_ARRAY_ARGUMENT_NOT_ALLOWED"
    return None


def candidate_module_policy_error(
    source: str, *, allowed_entrypoints: tuple[str, ...] = ()
) -> str | None:
    """Return the first safe-language policy error, or ``None`` when admitted."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return "CANDIDATE_SYNTAX_INVALID"
    top_level = set(tree.body)
    for node in tree.body:
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
            if isinstance(node.value.value, str):
                continue
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            if _import_key(node) in _ALLOWED_IMPORTS:
                continue
            return "CANDIDATE_IMPORT_NOT_ALLOWED"
        if isinstance(node, ast.FunctionDef):
            if not _function_signature_is_static(node):
                return "CANDIDATE_FUNCTION_SIGNATURE_NOT_STATIC"
            continue
        if isinstance(node, ast.Assign) and _constant_targets(node):
            continue
        return f"CANDIDATE_TOP_LEVEL_NODE_NOT_ALLOWED:{type(node).__name__}"
    workload_error = _workload_normal_form_error(
        tree, allowed_entrypoints=allowed_entrypoints
    )
    if workload_error is not None:
        return workload_error
    for node in ast.walk(tree):
        if isinstance(node, _FORBIDDEN_NODES):
            return f"CANDIDATE_NODE_NOT_ALLOWED:{type(node).__name__}"
        if isinstance(node, (ast.Import, ast.ImportFrom)) and node not in top_level:
            return "CANDIDATE_NESTED_IMPORT_NOT_ALLOWED"
        if isinstance(node, ast.FunctionDef) and node not in top_level:
            if not _function_signature_is_static(node):
                decorators = node.decorator_list
                if not (
                    len(decorators) == 1
                    and isinstance(decorators[0], ast.Call)
                    and _call_path(decorators[0]) == "pl.when"
                    and not node.args.defaults
                    and not any(
                        default is not None for default in node.args.kw_defaults
                    )
                ):
                    return "CANDIDATE_NESTED_FUNCTION_NOT_STATIC"
        if isinstance(node, ast.FunctionDef) and node.name in _RESERVED_BINDINGS:
            return "CANDIDATE_RESERVED_NAME_REBIND_NOT_ALLOWED"
        if isinstance(node, ast.arg) and node.arg in _RESERVED_BINDINGS:
            return "CANDIDATE_RESERVED_NAME_REBIND_NOT_ALLOWED"
        if isinstance(node, ast.Name) and node.id.startswith("__"):
            return "CANDIDATE_DUNDER_ACCESS_NOT_ALLOWED"
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            return "CANDIDATE_DUNDER_ACCESS_NOT_ALLOWED"
        if (
            isinstance(node, ast.Attribute)
            and node.attr in _FORBIDDEN_CAPABILITY_ATTRIBUTES
        ):
            return f"CANDIDATE_CAPABILITY_NOT_ALLOWED:{node.attr}"
        if isinstance(node, ast.Call) and not _call_is_numeric_or_pallas(
            node, allowed_entrypoints=allowed_entrypoints
        ):
            return f"CANDIDATE_CALL_NOT_ALLOWED:{_call_path(node) or 'dynamic'}"
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign, ast.NamedExpr)):
            targets = list(node.targets) if isinstance(node, ast.Assign) else [node.target]
            target_error = _binding_target_error(targets, allow_ref_subscript=True)
            if target_error is not None:
                return target_error
        if isinstance(node, ast.For):
            target_error = _binding_target_error(
                [node.target], allow_ref_subscript=False
            )
            if target_error is not None:
                return target_error
        if isinstance(node, ast.comprehension):
            target_error = _binding_target_error(
                [node.target], allow_ref_subscript=False
            )
            if target_error is not None:
                return target_error
    return None
