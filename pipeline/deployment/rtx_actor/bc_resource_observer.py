"""Read-only BC cgroup observation, pinned to one Docker/process lifetime.

Docker is used to bind ownership once, not polled on every memory sample.
Unavailable reads are distinct from proven unsafe counters or changed owners.
No workload submission, restart, signal or compute management lives here.
"""
import json
from pathlib import Path
import re
import subprocess
import time


class ObservationUnavailable(RuntimeError):
    pass


class OwnerChanged(RuntimeError):
    pass


def observe_command(process, read_live, read_terminal, terminal_exists):
    """Reconcile a disappearing cgroup with a completed command, not a replay.

    Docker's attached client can lag its exited native PID. Only a fully
    validated terminal receipt permits a bounded wait for that exact client's
    zero exit. Both receipt reads must agree. A receipt alone, a dead PID alone,
    nonzero exit or changed owner is NEVER successful completion.
    """
    code = process.poll()
    if code is not None:
        if code != 0:
            raise RuntimeError('exact native launcher exited nonzero')
        return dict(lifecycle='completed_attested', exit_code=0, **read_terminal())
    try:
        return read_live()
    except ObservationUnavailable:
        if not terminal_exists():
            raise
        before = read_terminal()  # Validates request/source/reset/all-window facts.
        try:
            code = process.wait(timeout=2)
        except subprocess.TimeoutExpired as exc:
            raise ObservationUnavailable('validated terminal receipt; exact launcher exit still pending') from exc
        if code != 0:
            raise RuntimeError('terminal receipt exists but exact launcher exited nonzero')
        after = read_terminal()
        if before != after:
            raise OwnerChanged('terminal evidence changed while awaiting exact launcher exit')
        return dict(lifecycle='completed_attested', exit_code=0, **after)


def wait_for_observer(root=Path('/dagger-runtime'), *, delay=time.sleep, clock=time.monotonic):
    """Native cooperative hold at step/writer boundaries, never a reset/replay."""
    marker = Path(root) / 'observer-hold.json'
    started = clock()
    while marker.exists():
        if marker.is_symlink():
            raise RuntimeError('observer hold must be an owned regular marker')
        if clock() - started >= 30:
            raise ObservationUnavailable('observer hold exceeded30seconds; preserve incomplete work')
        delay(.1)


def attest_scope_limits(group, cap_bytes, *, run=subprocess.run):
    """Read-only creation-policy check; no daemon reload or set-property."""
    if not re.fullmatch(r'/system.slice/docker-[0-9a-f]{64}\.scope', group):
        raise OwnerChanged('exact Docker systemd scope required')
    try:
        result = run(['systemctl', 'show', Path(group).name,
                      '--property=ControlGroup,MemoryMax,MemorySwapMax'],
                     capture_output=True, text=True, timeout=3)
    except (OSError, subprocess.SubprocessError) as exc:
        raise ObservationUnavailable('systemd scope observation unavailable: ' + type(exc).__name__) from exc
    if result.returncode or len(result.stdout) > 4096:
        raise ObservationUnavailable('systemd scope properties unavailable')
    rows = result.stdout.splitlines()
    parsed = dict(line.split('=', 1) for line in rows if '=' in line)
    expected = dict(ControlGroup=group, MemoryMax=str(cap_bytes), MemorySwapMax='0')
    if len(rows) != 3 or len(parsed) != 3 or parsed != expected:
        raise OwnerChanged('persistent systemd memory/no-swap properties differ: ' +
                           json.dumps({k: parsed.get(k) for k in expected}, sort_keys=True))
    return parsed


class PinnedCgroupObserver:
    def __init__(self, cid_path, name, image, cap_bytes, *, proc_root=Path('/proc'),
                 cgroup_root=Path('/sys/fs/cgroup'), inspect=None, scope_check=attest_scope_limits):
        self.proc_root, self.cgroup_root = Path(proc_root), Path(cgroup_root)
        try:
            cid_path = Path(cid_path)
            if cid_path.is_symlink():
                raise OwnerChanged('cidfile is a symlink')
            self.cid = cid_path.read_text().strip()
            # docker run creates the cidfile before container creation finishes.
            # Empty is pre-acceptance UNKNOWN, not a changed accepted identity.
            if not self.cid:
                raise ObservationUnavailable('Docker cidfile created; acceptance ID not written yet')
            if not re.fullmatch('[0-9a-f]{64}', self.cid):
                raise OwnerChanged('invalid Docker acceptance ID')
            if inspect is None:
                info = json.loads(subprocess.check_output(
                    ['docker', 'inspect', '--type', 'container', self.cid],
                    stderr=subprocess.PIPE, timeout=3))[0]
            else:
                info = inspect(self.cid)
            if (info['Id'] != self.cid or info['Name'] != '/' + name or info['Image'] != image
                    or info['Config']['Labels'].get('org.roborender.qualification') != name
                    or info['HostConfig']['Memory'] != cap_bytes
                    or info['HostConfig']['MemorySwap'] != cap_bytes):
                raise OwnerChanged('Docker owner/image/memory binding changed')
            if not info['State']['Running'] or not info['State']['Pid']:
                raise ObservationUnavailable('accepted Docker process is not running')
            self.pid = info['State']['Pid']
            self.name, self.cap_bytes = name, cap_bytes
            self.boot_id = self._boot()
            self.start_ticks = self._start()
            self.group = self._group()
            self.path = self.cgroup_root / self.group.lstrip('/')
            self.inode = self._inode()
            self.scope_properties = scope_check(self.group, self.cap_bytes)
        except (OSError, ValueError, KeyError, IndexError, subprocess.SubprocessError) as exc:
            raise ObservationUnavailable(f'{type(exc).__name__}: {exc}') from exc

    def _boot(self):
        return (self.proc_root / 'sys/kernel/random/boot_id').read_text().strip()

    def _start(self):
        raw = (self.proc_root / str(self.pid) / 'stat').read_text()
        fields = raw[raw.rindex(')') + 2:].split()
        if fields[0] in ('Z', 'X'):
            raise ObservationUnavailable('native process has exited; reconcile terminal receipt')
        return int(fields[19])

    def _group(self):
        entries = [line[3:] for line in (self.proc_root / str(self.pid) / 'cgroup').read_text().splitlines()
                   if line.startswith('0::')]
        if len(entries) != 1 or not entries[0].startswith('/') or '..' in Path(entries[0]).parts:
            raise OwnerChanged('native cgroup mapping is not canonical v2')
        return entries[0]

    def _inode(self):
        stat = self.path.stat()
        return stat.st_dev, stat.st_ino

    def _identity(self):
        if (self._boot() != self.boot_id or self._start() != self.start_ticks
                or self._group() != self.group or self._inode() != self.inode):
            raise OwnerChanged('native boot/PID start/cgroup identity changed; never rebind')

    def sample(self):
        try:
            self._identity()
            value = dict(container=self.name, container_id=self.cid, pid=self.pid,
                process_start_ticks=self.start_ticks, boot_id=self.boot_id,
                cgroup=self.group, cgroup_inode=list(self.inode), oom_killed=False,
                persistent_scope_properties=self.scope_properties,
                observation_source='pinned_host_cgroup_v1')
            for field in ('current', 'peak', 'max', 'swap.current', 'swap.max'):
                raw = (self.path / ('memory.' + field)).read_text().strip()
                # "max" is a valid kernel limit, not a transient parse failure.
                # Preserve the exact changed field and partial sample in evidence.
                if field in ('max', 'swap.max') and raw == 'max':
                    value['memory_' + field] = raw
                    self._identity()
                    raise OwnerChanged('actual cgroup limit became unlimited: ' + json.dumps(value, sort_keys=True))
                try:
                    value['memory_' + field] = int(raw)
                except ValueError as exc:
                    raise ObservationUnavailable('invalid memory.' + field + ' counter: ' + repr(raw)) from exc
            for field in ('events', 'stat'):
                pairs = dict(line.split() for line in (self.path / ('memory.' + field)).read_text().splitlines())
                value[field] = {k: int(v) for k, v in pairs.items()
                    if field == 'events' or k in ('anon', 'file', 'kernel', 'shmem', 'file_dirty', 'file_writeback')}
            self._identity()
            if value['memory_max'] != self.cap_bytes or value['memory_swap.max'] != 0:
                raise OwnerChanged('actual cgroup memory/no-swap limits changed: ' + json.dumps(value, sort_keys=True))
            return value
        except (OSError, ValueError, KeyError, IndexError) as exc:
            raise ObservationUnavailable(f'{type(exc).__name__}: {exc}') from exc


def recover_observation(read, validate, *, journal, hold, release, host_check,
                        delay=time.sleep, attempts=3, clock=time.monotonic):
    """Bounded read-only attempts; unknown pauses native progress.

    Every attempt is durable before validation. Proven unsafe values/changed
    ownership never retry. Persistent UNKNOWN remains held for caller cleanup.
    No stale sample can count as a recovered healthy read.
    Initial Docker acceptance may use up to20 attempts; bound owners use3.
    """
    if type(attempts) is not int or not 1 <= attempts <= 20:
        raise ValueError('observation attempts must be bounded to1..20')
    held = False
    deadline = clock() + 12
    for attempt in range(1, attempts + 1):
        if clock() >= deadline:
            journal(dict(status='UNKNOWN_DEADLINE', attempt=attempt))
            raise ObservationUnavailable('fresh observation deadline exceeded; never use stale counters')
        host_check()
        try:
            value = read()
        except ObservationUnavailable as exc:
            journal(dict(status='UNKNOWN', attempt=attempt, error_type=type(exc).__name__, error=str(exc)))
            if not held:
                hold()
                held = True
            if attempt == attempts:
                raise ObservationUnavailable(f'{attempts} failed fresh read attempts; native remains held') from exc
            delay(.5)
            continue
        except Exception as exc:
            journal(dict(status='OWNER_OR_READER_FAILURE', attempt=attempt,
                         error_type=type(exc).__name__, error=str(exc)))
            raise
        journal(dict(status='OBSERVED', attempt=attempt, sample=value))
        try:
            validate(value)
        except Exception as exc:
            journal(dict(status='REJECTED', attempt=attempt, error_type=type(exc).__name__, error=str(exc)))
            raise
        host_check()
        if held:
            release()
            journal(dict(status='RECOVERED', attempt=attempt))
        return value
