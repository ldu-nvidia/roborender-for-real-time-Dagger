"""Opt-in process-local THP control, never a host-wide memory-policy change.

Linux PR_SET_THP_DISABLE is inherited by child processes. Existing huge pages
are not split by this helper. Default callers perform no prctl at all.
Reference: https://docs.kernel.org/admin-guide/mm/transhuge.html#process-thp-controls
"""
import ctypes
import os
import sys

INHERITED = 'inherited'
DISABLED = 'disable-v1'
ENV = 'RESIDENT_THP_POLICY'
PR_SET_THP_DISABLE = 41
PR_GET_THP_DISABLE = 42


def validate_policy(value):
    if value not in (INHERITED, DISABLED):
        raise ValueError('unknown explicit native THP policy')
    return value


def _prctl(option, value=0):
    if sys.platform != 'linux':
        raise RuntimeError('native process THP control requires Linux')
    call = ctypes.CDLL(None, use_errno=True).prctl
    call.argtypes = [ctypes.c_int, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong]
    call.restype = ctypes.c_int
    result = call(option, value, 0, 0, 0)
    if result == -1:
        raise OSError(ctypes.get_errno(), 'native process THP prctl failed')
    return result


def configure(policy, *, prctl=None):
    validate_policy(policy)
    result = dict(schema='roborender.native-thp-policy.v1', policy=policy,
                  native_pid=os.getpid(), global_host_policy_modified=False,
                  existing_huge_pages_split=False)
    if policy == INHERITED:
        return result
    call = _prctl if prctl is None else prctl
    before = call(PR_GET_THP_DISABLE, 0)
    if before not in (0, 1):
        raise RuntimeError('unexpected preexisting native THP state')
    if before == 0 and call(PR_SET_THP_DISABLE, 1) != 0:
        raise RuntimeError('native THP disable failed')
    after = call(PR_GET_THP_DISABLE, 0)
    if after != 1:
        raise RuntimeError('native THP setting was not applied')
    return dict(result, disabled_before=before, disabled_after=after)


def configure_from_environment():
    return configure(os.environ.get(ENV, INHERITED))


def attest(receipt, *, prctl=None):
    """Read actual state again; never silently reapply a changed setting."""
    if receipt['native_pid'] != os.getpid():
        raise RuntimeError('native THP receipt belongs to another process')
    policy = validate_policy(receipt['policy'])
    if policy == DISABLED:
        call = _prctl if prctl is None else prctl
        if receipt.get('disabled_after') != 1 or call(PR_GET_THP_DISABLE, 0) != 1:
            raise RuntimeError('native THP state drifted')
    return dict(receipt)
