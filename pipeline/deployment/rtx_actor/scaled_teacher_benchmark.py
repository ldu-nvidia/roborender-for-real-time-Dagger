"""Compatibility import; canonical implementation is in pipeline.success_rate."""
import importlib as _importlib
import sys as _sys
_impl = _importlib.import_module("pipeline.success_rate.scaled_teacher_benchmark")
_sys.modules[__name__] = _impl
globals().update(vars(_impl))
