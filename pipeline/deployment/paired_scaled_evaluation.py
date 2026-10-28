"""Compatibility import only; implementation: pipeline.success_rate.paired_scaled_evaluation."""
import importlib as _importlib
import sys as _sys
_impl = _importlib.import_module("pipeline.success_rate.paired_scaled_evaluation")
_sys.modules[__name__] = _impl
globals().update(vars(_impl))
