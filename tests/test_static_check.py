from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "static_check", ROOT / "scripts" / "static_check.py",
)
static_check = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(static_check)


def test_static_check_detects_corrupted_telemetry_name(tmp_path):
    broken = tmp_path / "broken.py"
    broken.write_text(
        "def f():\n"
        "    _safe__runtime.record_repair_outcome()\n",
        encoding="utf-8",
    )
    problems = static_check.check_source(broken)
    assert any("forbidden corrupted name" in value for value in problems)
    assert any("undefined global name '_safe__runtime'" in value for value in problems)


def test_static_check_detects_shadowed_module_helper(tmp_path):
    broken = tmp_path / "shadow.py"
    broken.write_text(
        "def _model_number(value):\n"
        "    return int(value)\n"
        "\n"
        "def recover(affected):\n"
        "    _model_number, worker = affected[0], {}\n"
        "    return _model_number(worker, 0)\n",
        encoding="utf-8",
    )
    problems = static_check.check_source(broken)
    assert any("shadows a module-level callable" in value for value in problems)
