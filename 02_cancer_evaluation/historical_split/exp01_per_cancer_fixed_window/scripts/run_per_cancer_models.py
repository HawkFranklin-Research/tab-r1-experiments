from __future__ import annotations

import argparse
import sys
from pathlib import Path

SHARED = Path(__file__).resolve().parents[2] / "shared/scripts"
if str(SHARED) not in sys.path:
    sys.path.insert(0, str(SHARED))

from run_foundation_models import main as shared_main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(shared_main())
