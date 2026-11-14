"""Tutorial-only Python startup customization.

This project is frequently run on Windows with scientific Python stacks that may
load multiple OpenMP runtimes (e.g., MKL + another library). In some setups this
can trigger Intel OpenMP Error #15 ("libiomp5md.dll already initialized") and
abort execution before the tutorial scripts run.

Workaround:
  - Set KMP_DUPLICATE_LIB_OK=TRUE if it is not already set.

Notes:
  - This file is intentionally scoped to `tutorial/theory/` so it only affects
    tutorial scripts (Python automatically imports `sitecustomize` at startup
    when present on sys.path; the script directory is the first entry).
  - This is an unsafe workaround; it prioritizes tutorial reproducibility over
    maximum OpenMP strictness.
"""

from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

