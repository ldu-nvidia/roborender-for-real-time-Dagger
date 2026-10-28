"""RTX-owned, fail-closed distributed DAgger runtime."""

__all__ = ["CONTRACT_VERSION", "RAVEN_VIEW_ORDER"]


def __getattr__(name):
    # CPU resource-policy readers must not import ndarray/model dependencies.
    # Preserve the original public constants for existing callers on demand.
    if name in __all__:
        from . import contracts
        return getattr(contracts,name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
