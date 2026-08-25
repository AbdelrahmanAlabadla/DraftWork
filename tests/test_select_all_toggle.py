"""Runs the real frontend toggle logic (FrontEnd/js/select_all.js) via Node.

The pure function computeSelectAllLabel drives the Select/Unselect All button
label; these tests execute the ACTUAL frontend code, not a Python copy.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

NODE = shutil.which("node")
JS_FILE = Path(__file__).resolve().parent.parent / "FrontEnd" / "js" / "select_all.js"

pytestmark = pytest.mark.skipif(NODE is None, reason="node not available")

_SCRIPT = """
const { pathToFileURL } = require('url');
(async () => {
  const { computeSelectAllLabel } = await import(pathToFileURL(process.env.JS_FILE).href);
  const cases = JSON.parse(process.env.CASES);
  const out = [];
  for (const c of cases) {
    const got = computeSelectAllLabel(c.selected, c.total);
    out.push({ name: c.name, pass: got === c.expected, got });
  }
  console.log(JSON.stringify(out));
})();
"""

CASES = [
    # None selected -> Select All
    {"name": "none_selected", "selected": 0, "total": 10, "expected": "Select All Titles"},
    # Partial selection -> Select All
    {"name": "partial_selected", "selected": 4, "total": 10, "expected": "Select All Titles"},
    # All selected -> Unselect All
    {"name": "all_selected", "selected": 10, "total": 10, "expected": "Unselect All Titles"},
    # One deselected from all -> back to Select All
    {"name": "one_deselected", "selected": 9, "total": 10, "expected": "Select All Titles"},
    # Empty document -> Select All (button hidden anyway)
    {"name": "empty", "selected": 0, "total": 0, "expected": "Select All Titles"},
]


@pytest.fixture()
def mjs_copy(tmp_path):
    """Node treats bare .js under -e as CJS; copy to .mjs to import as ESM."""
    dest = tmp_path / "select_all.mjs"
    shutil.copy(JS_FILE, dest)
    return str(dest)


@pytest.mark.parametrize("case", CASES, ids=[c["name"] for c in CASES])
def test_toggle_label(case, mjs_copy):
    proc = subprocess.run(
        [NODE, "-e", _SCRIPT],
        capture_output=True, text=True, timeout=30,
        env={**os.environ, "JS_FILE": mjs_copy, "CASES": json.dumps([case])},
    )
    assert proc.returncode == 0, proc.stderr
    results = json.loads(proc.stdout)
    assert results[0]["pass"], (
        f"{case['name']}: expected '{case['expected']}' got '{results[0]['got']}'"
    )
