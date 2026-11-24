"""LongMemEval pinned harness adapter (oracle, temporal-reasoning slice).

Exports are resolved lazily so that ``python -m harnesses.longmemeval.adapter``
does not pre-import the adapter as a second module instance.
"""

__all__ = ["LANE", "JUDGE", "prepare", "select", "evaluate", "self_test"]

_ADAPTER = None


def _load_adapter():
    global _ADAPTER
    if _ADAPTER is None:
        try:  # normal package import
            from . import adapter as _ADAPTER  # type: ignore[misc]
        except ImportError:  # loaded directly by path: load sibling explicitly
            import importlib.util as _ilu
            import sys
            from pathlib import Path

            _dir = Path(__file__).resolve().parent
            sys.path.insert(0, str(_dir.parent))  # make harnesses.common importable
            _spec = _ilu.spec_from_file_location(
                "_memnoth_bench_longmemeval_adapter", _dir / "adapter.py"
            )
            _ADAPTER = _ilu.module_from_spec(_spec)
            sys.modules[_spec.name] = _ADAPTER
            _spec.loader.exec_module(_ADAPTER)
    return _ADAPTER


def __getattr__(name):
    if name in __all__:
        return getattr(_load_adapter(), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
