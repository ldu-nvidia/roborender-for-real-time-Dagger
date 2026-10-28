"""Explicit B32x2 diagnostic: shared stepping leases, exclusive CPU save lease.

No reset/action/chunk changes. The default single-collector guard stays at8GiB.
Only a fresh, hash-bound paired smoke may use the user-authorized4GiB reserve.
This does not certify production or provide native replay after a failure.
"""
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time

GIB = 2**30
SCHEMA = 'roborender.bc-paired32-smoke.v1'


def validate_config(value):
    if (value.get('schema') != SCHEMA or value.get('num_envs_per_actor') != 32
            or value.get('windows_per_actor') != 4 or value.get('host_reserve_bytes') != 4 * GIB
            or value.get('actor_cap_bytes') != 43 * GIB
            or value.get('production_admitted') is not False
            or value.get('retain_compute') is not True or value.get('replay_allowed') is not False
            or value.get('seeds') != [0, 1]
            or len(value.get('request_sha256', [])) != 2
            or any(not isinstance(s, str) or len(s) != 64 or any(c not in '0123456789abcdef' for c in s)
                   for s in [*value.get('request_sha256', []), value.get('source_sha256'),
                             value.get('guard_sha256')])):
        raise ValueError('explicit B32x2 four-window four-GiB diagnostic binding required')
    return value


class PairLease:
    def __init__(self, root, actor, config):
        self.root, self.actor, self.config = Path(root), actor, validate_config(config)
        if type(actor) is not int or actor not in (0, 1):
            raise ValueError('exact paired actor required')
        self.fd = os.open(self.root / 'save.lock', os.O_RDWR | os.O_NOFOLLOW)
        self.held = False

    def check(self):
        started = time.monotonic()
        while True:
            if (self.root / 'abort.json').exists():
                raise RuntimeError('paired controller aborted; no native retry')
            if not (self.root / 'observation-hold.json').exists():
                return
            if time.monotonic() - started > 30:
                raise TimeoutError('resource observation unresolved for30seconds; no native replay')
            time.sleep(.1)

    def event(self, phase, **fields):
        with (self.root / f'actor{self.actor}-events.jsonl').open('a') as stream:
            stream.write(json.dumps(dict(phase=phase, actor=self.actor, time=time.monotonic(), **fields)) + '\n')
            stream.flush()
            os.fsync(stream.fileno())

    def ready(self):
        row = dict(actor=self.actor, pid=os.getpid(), request_sha256=self.config['request_sha256'][self.actor])
        target = self.root / f'ready{self.actor}.json'
        temporary = self.root / f'ready{self.actor}.part'
        with temporary.open('x') as stream:
            json.dump(row, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, target)
        temporary.unlink()
        start = time.monotonic()
        while True:
            self.check()
            if all((self.root / f'ready{i}.json').exists() for i in (0, 1)):
                for i in (0, 1):
                    value = json.loads((self.root / f'ready{i}.json').read_bytes())
                    if value.get('actor') != i or value.get('request_sha256') != self.config['request_sha256'][i]:
                        raise RuntimeError('peer startup identity changed')
                return
            if time.monotonic() - start > 1200:
                raise TimeoutError('peer startup not ready; never step alone')
            time.sleep(.2)

    def acquire(self, exclusive=False):
        if self.held:
            raise RuntimeError('nested paired lease')
        started = time.monotonic()
        while True:
            self.check()
            waiting = not exclusive and any((self.root / f'save-waiting{i}').exists() for i in (0, 1))
            if not waiting:
                try:
                    fcntl.flock(self.fd, (fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH) | fcntl.LOCK_NB)
                    self.held = True
                    return
                except BlockingIOError:
                    pass
            if time.monotonic() - started > 1200:
                raise TimeoutError('paired save/step lease stalled; no replay')
            time.sleep(.05)

    def release(self):
        if self.held:
            fcntl.flock(self.fd, fcntl.LOCK_UN)
            self.held = False

    @contextmanager
    def saving(self):
        pending = self.root / f'save-waiting{self.actor}'
        pending.touch(exist_ok=False)
        try:
            self.acquire(exclusive=True)
            self.event('save_begin')
            try:
                yield
            finally:
                self.event('save_end')
        finally:
            self.release()
            pending.unlink()

    def close(self):
        self.release()
        os.close(self.fd)


def __getattr__(name):
    if name != 'PairedBCTrainer':
        raise AttributeError(name)
    import bounded_bc_guard
    from isaac6_bc_session import memory_snapshot
    from isaac6_bc_production import load_request

    raw = Path('/bc-pair/config.json').read_bytes()
    if hashlib.sha256(raw).hexdigest() != os.environ.get('ROBORENDER_PAIR_CONFIG_SHA256'):
        raise RuntimeError('paired configuration bytes changed')
    config = validate_config(json.loads(raw))
    if hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != config['guard_sha256']:
        raise RuntimeError('paired guard source changed')
    actor = int(os.environ['ROBORENDER_PAIR_ACTOR'])
    request = load_request()
    if (actor not in (0, 1) or len(request['motions']) != 32 or request['max_windows'] != 4
            or request['process_seed'] != actor or 'diagnostic_stop' in request
            or os.environ['ROBORENDER_BC_COLLECTION_REQUEST_SHA256'] != config['request_sha256'][actor]):
        raise RuntimeError('paired collector request changed')

    def available():
        rows = dict(line.split(':', 1) for line in Path('/run/host-meminfo').read_text().splitlines())
        return int(rows['MemAvailable'].split()[0]) * 1024

    class PairedBCTrainer(bounded_bc_guard.BoundedContinuousBCTrainer):
        def train(self):
            self._pair = PairLease('/bc-pair', actor, config)
            self._pair_saving, self._pair_ready, self._pair_steps = False, False, 0
            original_step = self.env.step

            def step(*args, **kwargs):
                try:
                    result = original_step(*args, **kwargs)
                    self._pair_steps += 1
                    if self._pair_steps % 81 == 0:
                        self._pair.event('capture_end', steps=self._pair_steps)
                    return result
                finally:
                    self._pair.release()

            self.env.step = step
            try:
                return super().train()
            finally:
                self.env.step = original_step
                self._pair.close()

        def policy_step(self, *args, **kwargs):
            if not self._pair_ready:
                self._pair.ready()
                self._pair_ready = True
            self._pair.acquire()
            try:
                if self._pair_steps % 81 == 0:
                    self._pair.event('capture_begin', steps=self._pair_steps)
                return super().policy_step(*args, **kwargs)
            except BaseException:
                self._pair.release()
                raise

        def _check_host_memory(self, *, saving=False, tail_growth_bytes=None):
            self._pair.check()
            sample, free = memory_snapshot(), available()
            try:
                if sample.get('events', {}).get('max') != 0:
                    raise RuntimeError('paired cgroup limit pressure is not admitted')
                return bounded_bc_guard.memory_gate(sample, free, num_envs=32, cap_bytes=43 * GIB,
                    saving=saving, tail_growth_bytes=tail_growth_bytes, host_reserve_bytes=4 * GIB,
                    peer_capture_bytes=0 if self._pair_saving else 2 * GIB)
            except BaseException as exc:
                self._pair.event('memory_rejected', memory=sample, host_available_bytes=free,
                                 saving=saving, tail_growth_bytes=tail_growth_bytes, error=str(exc))
                raise

        def _save_collected_chunks(self):
            with self._pair.saving():
                self._pair_saving = True
                try:
                    return super()._save_collected_chunks()
                finally:
                    self._pair_saving = False

    globals()[name] = PairedBCTrainer
    return PairedBCTrainer
