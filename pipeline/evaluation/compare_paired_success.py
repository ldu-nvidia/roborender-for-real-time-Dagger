"""Compatibility import only; implementation: pipeline.success_rate.compare_paired_success."""
import importlib as _importlib
import sys as _sys
_impl = _importlib.import_module("pipeline.success_rate.compare_paired_success")
if __name__ == "__main__":
    raise SystemExit(_impl.main())
_sys.modules[__name__] = _impl
globals().update(vars(_impl))
