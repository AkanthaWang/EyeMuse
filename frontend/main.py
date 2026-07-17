from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SRC = Path(__file__).resolve().parent / "src"

for path in (ROOT, SRC):
    path_text = str(path)
    if path_text not in sys.path:
        sys.path.insert(0, path_text)

from app import run  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(run())