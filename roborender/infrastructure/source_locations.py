"""Exact Leo source-location admission, independent of storage aliases.

The storage registry describes locations; it cannot authorize another executable
root. These two physical release jails are intentionally static. New launcher
source supports the canonical location without modifying any frozen release.
Location admission never substitutes for a source manifest/hash check.
"""
from pathlib import Path


LEGACY_RELEASE_ROOT = Path('/home/ldu/roborender/runtime/source-releases')
CANONICAL_RELEASE_ROOT = Path('/data1/workspaces/roborender/code_releases')
LEGACY_RUNTIME_ROOT = Path('/home/ldu/roborender/runtime')


def is_source_release_location(value):
    """Lexical remote-host preflight only; the launcher must resolve locally.

    Retain the original legacy prefix admission. The NEW jail permits only a
    release's code/frozen tree and descendants, not its collection, an arbitrary
    sibling or mutable checkout. No registry, environment or caller root widens
    either jail; manifests still establish the executable source identity.
    """
    try:
        path = Path(value)
    except TypeError:
        return False
    if not path.is_absolute() or '..' in path.parts or '\x00' in str(path):
        return False
    if path.is_relative_to(LEGACY_RELEASE_ROOT):
        return True
    if path.is_relative_to(CANONICAL_RELEASE_ROOT):
        relative = path.relative_to(CANONICAL_RELEASE_ROOT)
        if len(relative.parts) >= 2 and relative.parts[1] in ('code', 'frozen'):
            return True
    return False


def resolve_source_release(value, *, error='immutable source release required'):
    """Return an existing physical code directory inside an exact release jail.

    Resolve the candidate, not the allowed root: redirecting an allowed-root
    symlink outside its physical jail must not grant access to the new target.
    """
    path = Path(value).resolve(strict=True)
    if not is_source_release_location(path) or not path.is_dir():
        raise ValueError(error)
    return path


def resolve_native_code(value, *, error='source must be a scoped runtime or immutable source release'):
    """Retain old diagnostic runtime admission; add only canonical releases.

    Two historical diagnostic launchers accepted code below the legacy runtime,
    not only source-releases. Keep that compatibility without admitting any new
    canonical runs/caches/checkouts or resolving the legacy jail to another root.
    """
    path = Path(value).resolve(strict=True)
    legacy = path.is_relative_to(LEGACY_RUNTIME_ROOT)
    if not path.is_dir() or not (legacy or is_source_release_location(path)):
        raise ValueError(error)
    return path
