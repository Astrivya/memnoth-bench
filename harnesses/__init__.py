"""Pinned harness adapters for external memory benchmarks.

Stdlib-only. Import-safe: importing this package never downloads a dataset,
never reads a key, and never performs spend. All dataset I/O happens only when
an adapter's ``prepare()`` is called explicitly.
"""

try:  # normal package import
    from . import common
    from .common import PIN_BEFORE_USE
except ImportError:  # loaded directly by path (e.g. importlib.util.spec_from_file_location)
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import common
    from common import PIN_BEFORE_USE

__all__ = ["common", "PIN_BEFORE_USE"]
