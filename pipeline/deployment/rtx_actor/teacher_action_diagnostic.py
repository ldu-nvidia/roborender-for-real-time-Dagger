"""Compatibility import; implementation lives in success_rate.policy_actions."""
import importlib as _importlib
import sys as _sys
_impl = _importlib.import_module("pipeline.success_rate.policy_actions")
_sys.modules[__name__] = _impl
globals().update(vars(_impl))
