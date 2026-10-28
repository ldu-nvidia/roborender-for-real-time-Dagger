"""Compatibility import only; implementation: pipeline.success_rate.scaled_success_resources."""
import importlib as _importlib
import sys as _sys
_impl = _importlib.import_module("pipeline.success_rate.scaled_success_resources")
_sys.modules[__name__] = _impl
globals().update(vars(_impl))
