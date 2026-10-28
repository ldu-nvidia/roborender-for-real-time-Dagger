"""Compatibility import only; implementation: pipeline.success_rate.episode_dispatch."""
import importlib as _importlib
import sys as _sys
_impl = _importlib.import_module("pipeline.success_rate.episode_dispatch")
_sys.modules[__name__] = _impl
globals().update(vars(_impl))
