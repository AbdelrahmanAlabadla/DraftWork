"""Retired legacy 40-question harness.

The original test manually injected repair history without the corresponding
transactional candidate/lifecycle state and did not mock regeneration. Its
coverage is preserved and extended by test_draftwork_40q_readonly_e2e.py,
which exercises the current repair, regeneration, replacement, revalidation,
duplicate-review, and atomic commit paths with every provider boundary mocked.
"""

import pytest


pytest.skip(
    "retired: superseded by test_draftwork_40q_readonly_e2e.py",
    allow_module_level=True,
)
