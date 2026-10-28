"""Filesystem identity and reserve guards shared by BC collection launchers."""
from pathlib import Path
import os

STORAGE_SCHEMA = 'roborender.bc-campaign-storage-policy.v1'
GIB = 2**30
MIN_CHUNK_BUDGET = 512 << 20


class StorageRejected(RuntimeError):
    """No launch, or stop only the already-owned child; never replay a partial."""


def validate_storage_policy(policy):
    expected = {'schema', 'root', 'reserve_bytes', 'bytes_per_emission_budget',
                'session_metadata_budget_bytes', 'poll_seconds'}
    if (not isinstance(policy, dict) or set(policy) != expected
            or policy['schema'] != STORAGE_SCHEMA
            or not isinstance(policy['root'], str) or not Path(policy['root']).is_absolute()
            or '..' in Path(policy['root']).parts or Path(policy['root']) == Path('/')
            or type(policy['reserve_bytes']) is not int or policy['reserve_bytes'] < 8 * GIB
            or type(policy['bytes_per_emission_budget']) is not int
            or policy['bytes_per_emission_budget'] < MIN_CHUNK_BUDGET
            or type(policy['session_metadata_budget_bytes']) is not int
            or policy['session_metadata_budget_bytes'] < GIB
            or type(policy['poll_seconds']) is not int or policy['poll_seconds'] != 1):
        raise ValueError('explicit bounded storage root/reserve/budget and one-second polling required')
    return policy


def storage_snapshot(policy, runtime, state_dir):
    root = Path(policy['root'])
    if root.resolve(strict=True) != root or not root.is_dir():
        raise StorageRejected('storage root is not a stable non-aliased directory')
    before = root.stat()
    for path in (Path(runtime), Path(state_dir)):
        if path.resolve(strict=False) != path or not path.is_relative_to(root):
            raise StorageRejected('owned output/state path escaped the guarded filesystem root')
        ancestor = path
        while not ancestor.exists():
            ancestor = ancestor.parent
        if ancestor.stat().st_dev != before.st_dev:
            raise StorageRejected('owned output/state path is on an unguarded filesystem')
    value = os.statvfs(root)
    after = root.stat()
    if (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino) or value.f_frsize <= 0 or value.f_bavail < 0:
        raise StorageRejected('storage filesystem identity or free-space counters changed')
    return dict(root=str(root), device=after.st_dev, inode=after.st_ino,
                available_bytes=value.f_bavail * value.f_frsize)


def observe_storage(snapshot, policy, report):
    if (not isinstance(snapshot, dict) or type(snapshot.get('available_bytes')) is not int
            or snapshot['available_bytes'] < 0
            or any(type(snapshot.get(k)) is not int for k in ('device', 'inode'))
            or snapshot.get('root') != policy['root']):
        raise StorageRejected('storage counters unavailable or invalid')
    identity = {k: snapshot[k] for k in ('root', 'device', 'inode')}
    if 'filesystem' in report and report['filesystem'] != identity:
        raise StorageRejected('guarded filesystem identity changed during the owned session')
    report['filesystem'] = identity
    report['samples'] = report.get('samples', 0) + 1
    report['last_available_bytes'] = snapshot['available_bytes']
    report['minimum_available_bytes'] = min(report.get('minimum_available_bytes', snapshot['available_bytes']),
                                          snapshot['available_bytes'])
    if snapshot['available_bytes'] < policy['reserve_bytes']:
        raise StorageRejected('free storage fell below the declared reserve')
    return snapshot['available_bytes']
