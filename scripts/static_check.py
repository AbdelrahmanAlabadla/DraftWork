"""Static and import checks for the Validator refactor.

Catches the two classes of production crashes seen during the refactor:
* undefined names in rarely executed branches (``_safe__runtime``);
* local variables shadowing module-level helper functions (``_model_number``).

No third-party linter is required.  ``compileall`` plus ``symtable`` plus an AST
pass are deterministic and fast enough for the pre-deployment gate.
"""
from __future__ import annotations

import ast
import builtins
import compileall
import importlib
import symtable
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
APP = ROOT / "app"

CHECKED_FILES = [
    "app/online/validator.py",
    "app/online/validation/state.py",
    "app/online/validation/constants.py",
    "app/online/validation/runtime.py",
    "app/online/validation/questions.py",
    "app/online/validation/contracts.py",
    "app/online/validation/context.py",
    "app/online/validation/manifest.py",
    "app/online/validation/duplicate_coordinator.py",
    "app/online/validation/orchestrator.py",
    "app/online/validation/recovery.py",
    "app/online/validation/legacy/validation.py",
    "app/online/validation/legacy/repair.py",
    "app/online/validation/legacy/global_review.py",
    "app/online/independent_recovery.py",
    "app/online/duplicate_judge.py",
    "app/online/duplicate_pipeline.py",
    "app/online/multi_validator.py",
    "app/online/multi_repair.py",
    "app/online/cross_model_duplicates.py",
    "app/online/graph.py",
]

IMPORT_MODULES = [
    "app.online.validator",
    "app.online.validation.orchestrator",
    "app.online.validation.recovery",
    "app.online.independent_recovery",
    "app.online.multi_validator",
    "app.online.multi_repair",
    "app.online.duplicate_pipeline",
    "app.online.duplicate_judge",
    "app.online.cross_model_duplicates",
    "app.online.graph",
    "app.jobs.celery_app",
    "app.api.main",
]

FORBIDDEN_PATTERNS = ("_safe__runtime", "__runtime", "_runtime._runtime")


def _module_callables(tree: ast.Module) -> set[str]:
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                names.add(alias.asname or alias.name.split(".")[0])
    return names


def _walk_tables(table: symtable.SymbolTable):
    yield table
    for child in table.get_children():
        yield from _walk_tables(child)


def check_source(path: Path) -> list[str]:
    problems: list[str] = []
    source = path.read_text(encoding="utf-8")
    for pattern in FORBIDDEN_PATTERNS:
        if pattern in source:
            problems.append(f"{path}: forbidden corrupted name pattern {pattern!r}")
    tree = ast.parse(source, filename=str(path))
    module_callables = _module_callables(tree)
    table = symtable.symtable(source, str(path), "exec")
    module_names = {symbol.get_name() for symbol in table.get_symbols()}
    builtin_names = set(dir(builtins))
    for scope in _walk_tables(table):
        if scope.get_type() != "function":
            continue
        for symbol in scope.get_symbols():
            name = symbol.get_name()
            if symbol.is_local() and name in module_callables and name.startswith("_"):
                problems.append(
                    f"{path}:{scope.get_lineno()}: local {name!r} shadows a "
                    "module-level callable"
                )
            if symbol.is_global() and name not in module_names and name not in builtin_names:
                if name.startswith("__"):
                    continue
                problems.append(
                    f"{path}:{scope.get_lineno()}: undefined global name {name!r}"
                )
    return problems


def check_facade_bindings() -> list[str]:
    import app.online.validator as facade
    from app.online.validation import orchestrator, recovery

    problems = []
    if orchestrator.validate_generated_questions.__module__ != (
        "app.online.validation.orchestrator"
    ):
        problems.append("orchestrator implementation module mismatch")
    if recovery.repair_invalid_questions.__module__ != "app.online.validation.recovery":
        problems.append("recovery implementation module mismatch")
    for name in (
        "validate_generated_questions", "repair_invalid_questions",
        "validate_exam", "repair_exam", "persist_difficulty_observations",
        "_integrate_duplicate_validation",
    ):
        if not hasattr(facade, name):
            problems.append(f"validator facade missing {name}")
    return problems


def check_facade_imports() -> list[str]:
    import app.online.validator as facade

    problems = []
    for path in CHECKED_FILES:
        source = (ROOT / path).read_text(encoding="utf-8")
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom):
                continue
            if node.module != "app.online.validator":
                continue
            for alias in node.names:
                if not hasattr(facade, alias.name):
                    problems.append(
                        f"{path}:{node.lineno}: imports missing validator symbol "
                        f"{alias.name}"
                    )
    return problems


def main() -> int:
    problems: list[str] = []
    if not compileall.compile_dir(str(APP), quiet=1, force=True):
        problems.append("compileall failed")
    for module in IMPORT_MODULES:
        try:
            importlib.import_module(module)
        except Exception as exc:  # noqa: BLE001 - report any import failure
            problems.append(f"import failed: {module}: {type(exc).__name__}: {exc}")
    for relative in CHECKED_FILES:
        path = ROOT / relative
        if not path.exists():
            problems.append(f"missing file: {relative}")
            continue
        try:
            problems.extend(check_source(path))
        except SyntaxError as exc:
            problems.append(f"{relative}: syntax error: {exc}")
    problems.extend(check_facade_bindings())
    problems.extend(check_facade_imports())
    if problems:
        print("STATIC CHECK FAILED")
        for problem in problems:
            print(" -", problem)
        return 1
    print(f"STATIC CHECK OK ({len(CHECKED_FILES)} files, {len(IMPORT_MODULES)} imports)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
