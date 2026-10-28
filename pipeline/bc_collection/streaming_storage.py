"""Bounded per-window disk admission; never a whole-campaign fit guarantee.

The legacy whole-campaign512MiB/emission reservation remains unchanged. This
separate policy retains that per-emission ceiling, reserves a complete upcoming
vector-window flush plus metadata, and continuously checks real free space.
Exhaustion is a failed/incomplete run, NOT successful quota completion. No file
deletion, spill onto another filesystem, replay or compressed-mean assumption.
"""
from pathlib import Path
import os

PROTOCOL = 'roborender.bc-streaming-disk-guard.v1'
GIB = 2**30
EMISSION_BYTES = 512 << 20
RESERVE_BYTES = 64 * GIB
METADATA_BYTES = GIB


def policy(num_envs):
    if type(num_envs) is not int or not 1 <= num_envs <= 40:
        raise ValueError('explicit single-scene B1..40 required')
    return dict(protocol=PROTOCOL, num_envs=num_envs,
        bytes_per_emission_ceiling=EMISSION_BYTES, reserve_bytes=RESERVE_BYTES,
        metadata_bytes=METADATA_BYTES,
        window_admission_bytes=RESERVE_BYTES + num_envs * EMISSION_BYTES + METADATA_BYTES,
        whole_campaign_fit_guaranteed=False, shortage_is_success=False)


def archive_size_bound(values):
    """Conservative zlib/NPY/ZIP bound from actual arrays, not observed ratios.

    zlib compressBound is <1% above input for these arrays. Allow1% plus64KiB
    per NPY/ZIP member and1MiB for archive metadata. Explicitly reject object,
    oversized, unusual dtype and member-name inputs before serialization.
    This bound does not assume any compressibility of camera pixels.
    """
    import numpy as np
    if not isinstance(values, dict) or not 1 <= len(values) <= 128:
        raise ValueError('bounded explicit NPZ member mapping required')
    total = 1 << 20
    for name, value in values.items():
        if (not isinstance(name, str) or not name or len(name) > 128
                or not name.isascii() or '/' in name or '\\' in name
                or not isinstance(value, np.ndarray) or value.dtype.hasobject
                or value.dtype.fields is not None or value.ndim > 8):
            raise ValueError('unsupported NPZ member schema for size bound')
        total += value.nbytes + (value.nbytes + 99) // 100 + (64 << 10)
    if total > EMISSION_BYTES:
        raise RuntimeError('actual array serialization bound exceeds512MiB emission ceiling')
    return total


class StreamingDiskGuard:
    """Bind the actual mounted output directory, then check before work/write."""
    def __init__(self, root, num_envs):
        self.root = Path(root)
        if self.root.is_symlink() or self.root.resolve(strict=True) != self.root or not self.root.is_dir():
            raise ValueError('exact existing non-aliased native output directory required')
        self.policy = policy(num_envs)
        stat = self.root.stat()
        self.identity = stat.st_dev, stat.st_ino
        self.minimum_available_bytes = None
        self.windows_admitted = self.emissions_verified = 0

    def _free(self):
        before = self.root.stat()
        if (before.st_dev, before.st_ino) != self.identity or self.root.is_symlink():
            raise RuntimeError('streaming output filesystem/directory identity changed')
        stat = os.statvfs(self.root)
        after = self.root.stat()
        if ((after.st_dev, after.st_ino) != self.identity or stat.f_frsize <= 0 or stat.f_bavail < 0):
            raise RuntimeError('streaming output free-space observation unavailable')
        available = stat.f_bavail * stat.f_frsize
        self.minimum_available_bytes = min(self.minimum_available_bytes or available, available)
        return available

    def admit_window(self):
        # At most one81-row emission per lane per81 new vector steps, even with
        # an80-row previous tail. Native done discards, never pads a short tail.
        if self._free() < self.policy['window_admission_bytes']:
            raise RuntimeError('streaming disk headroom cannot admit the next full window; incomplete')
        self.windows_admitted += 1

    def admit_archive(self, path, values):
        path = Path(path)
        if path.parent != self.root or path.exists() or path.is_symlink() or path.suffix != '.npz':
            raise ValueError('fresh direct owned archive required')
        bound = archive_size_bound(values)
        if self._free() < RESERVE_BYTES + METADATA_BYTES + bound:
            raise RuntimeError('streaming disk headroom changed before archive; incomplete')
        return bound

    def verify_archive(self, path, bound):
        path = Path(path)
        if (path.parent != self.root or path.is_symlink() or not path.is_file()
                or not 0 < path.stat().st_size <= bound <= EMISSION_BYTES):
            raise RuntimeError('serialized BC archive exceeds its admitted byte bound')
        if self._free() < RESERVE_BYTES:
            raise RuntimeError('streaming disk reserve violated; incomplete')
        self.emissions_verified += 1

    def receipt(self):
        return dict(**self.policy, output_root=str(self.root), directory_identity=list(self.identity),
            minimum_available_bytes=self.minimum_available_bytes,
            windows_admitted=self.windows_admitted, emissions_verified=self.emissions_verified)
