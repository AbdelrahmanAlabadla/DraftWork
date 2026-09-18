from __future__ import annotations

import os


# Existing API regression tests exercise the old synchronous compatibility
# routes. Product/browser traffic uses the queued /api/v1 contract.
os.environ.setdefault("ENABLE_LEGACY_SYNC_API", "true")
