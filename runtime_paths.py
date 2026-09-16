"""Locate files shipped alongside the application when it is frozen.

PyInstaller places data files below ``sys._MEIPASS``.  During normal ``uv run``
development, the same files live beside this module in the repository.
"""

from __future__ import annotations

import sys
from pathlib import Path


def bundled_path(relative_path: str | Path) -> Path:
    """Return an absolute path to a repository or PyInstaller-bundled file."""
    root = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
    return root / relative_path
