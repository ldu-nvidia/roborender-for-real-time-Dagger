"""Small, actual cgroup/host samples for one bounded native production phase.

No model/data scans or compute ownership actions occur here. Missing counters
fail closed. The calling step guard checks the latched failure before every
transaction; there is no simulator replay after a failed phase.
"""
from __future__ import annotations

import hashlib
import math
import os
import re
import threading
import time
from pathlib import Path

GIB = 1 << 30
SCHEMA = 'roborender.native-phase-resource-guard.v2'
POLICY_VERSION = 'roborender.native-memory-policy.v2'
POLICY = dict(memory_limit_bytes=21*GIB, swap_limit_bytes=0,
              minimum_host_available_bytes=8*GIB, minimum_working_set_margin_bytes=GIB,
              maximum_some_avg10=10.0, maximum_full_avg10=5.0)
SUSTAINED_VERSION = 'roborender.native-memory-policy.v3'
SUSTAINED_SCHEMA = 'roborender.native-phase-resource-guard.v3'
SUSTAINED_POLICY = dict(POLICY, maximum_some_avg10=40.0, maximum_full_avg10=20.0,
    warning_some_avg10=10.0, warning_full_avg10=5.0, maximum_warning_seconds=30.0)


def policy_for(version):
    if version == POLICY_VERSION: return POLICY
    if version == SUSTAINED_VERSION: return SUSTAINED_POLICY
    raise RuntimeError('unknown native memory policy')


class SustainedPressureError(RuntimeError):
    """Public, bounded cause evidence; never serialize arbitrary exception text."""
    def __init__(self, window, value):
        super().__init__('sustained native memory PSI exceeded 30 seconds')
        self.diagnostic = dict(
            schema='roborender.native-pressure-failure.v1',
            reason_code='sustained_memory_pressure',
            policy_version=SUSTAINED_VERSION,
            threshold_seconds=SUSTAINED_POLICY['maximum_warning_seconds'],
            pressure_window=window.summary(),
            some_avg10=value['some_avg10'], full_avg10=value['full_avg10'])


class PressureWindow:
    """PSI is stall time, not an OOM counter. Keep hard memory gates immediate.

    v3 admits bounded moderate stalls, but never critical PSI or a >=30s
    continuous warning. No peak/counter is erased; v2 remains instantaneous.
    """
    def __init__(self):
        self.started = self.last = None
        self.warning_samples = 0
        self.maximum_seconds = 0.0
        self.failed = False

    def observe(self, value, now):
        if not math.isfinite(now) or (self.last is not None and now < self.last):
            raise RuntimeError('pressure observation clock regressed')
        self.last = now
        warning = (value['some_avg10'] > SUSTAINED_POLICY['warning_some_avg10']
                   or value['full_avg10'] > SUSTAINED_POLICY['warning_full_avg10'])
        if self.started is not None:
            self.maximum_seconds = max(self.maximum_seconds, now-self.started)
            # Even a recovered reading cannot erase an unobserved long interval.
            if now-self.started >= SUSTAINED_POLICY['maximum_warning_seconds']:
                self.failed = True
        if warning:
            self.warning_samples += 1
            if self.started is None: self.started = now
        else:
            self.started = None
        if self.failed: raise SustainedPressureError(self, value)

    def summary(self):
        return dict(warning_samples=self.warning_samples,
                    maximum_warning_seconds=self.maximum_seconds, failed=self.failed)
SUMMARY_FIELDS = ('schema','policy_version','minimum_host_available_bytes',
    'swap_peak_bytes','oom_events','limit_pressure_events','cache_reclaim_events',
    'unsafe_pressure_events','initial_limit_pressure_events','phase_limit_pressure_events',
    'phase_cache_reclaim_events','phase_unsafe_pressure_events','memory_limit_bytes',
    'swap_limit_bytes','maximum_working_set_bytes','maximum_some_avg10','maximum_full_avg10',
    'samples','passed')


def _integer(value):
    return type(value) is int and value >= 0


def _finite_percent(value):
    return type(value) in (float, int) and math.isfinite(value) and 0 <= value <= 100


def validate_identity(value):
    if (any(not _integer(value.get(k)) for k in ('process_id','cgroup_device','cgroup_inode'))
            or value['process_id'] == 0 or value['cgroup_inode'] == 0
            or not isinstance(value.get('cgroup_path'),str)
            or not Path(value['cgroup_path']).is_absolute()
            or '..' in Path(value['cgroup_path']).parts):
        raise RuntimeError('actual owned process/cgroup identity required')
    return tuple(value[k] for k in ('process_id','cgroup_path','cgroup_device','cgroup_inode'))


def validate_transition(previous, current):
    """One owned lifetime; event counters can increase but never regress."""
    identity = validate_identity(current)
    if previous is not None:
        if identity != validate_identity(previous):
            raise RuntimeError('owned native cgroup identity changed during the phase')
        if any(current['events'][key] < previous['events'][key] for key in ('max','oom','oom_kill')):
            raise RuntimeError('owned native event counters regressed during the phase')


def _validate_sample(value):
    keys = ('host_available_bytes', 'memory_current_bytes', 'memory_limit_bytes',
            'swap_bytes', 'swap_limit_bytes', 'inactive_file_bytes', 'working_set_bytes')
    if (not isinstance(value, dict) or any(not _integer(value.get(k)) for k in keys)
            or not isinstance(value.get('events'), dict)
            or any(not _integer(value['events'].get(k)) for k in ('max','oom','oom_kill'))
            or not all(_finite_percent(value.get(k)) for k in ('some_avg10','full_avg10'))):
        raise RuntimeError('invalid or unknown native v2 resource counters/PSI')
    validate_identity(value)
    return value


def classify_sample(value, *, policy_version=POLICY_VERSION):
    """Validate actual bounded memory, not an invented zero pressure counter.

    memory.events.max includes ordinary cgroup file-cache reclaim. A nonzero
    counter is safe only with the fixed 21GiB/zero-swap caps, >=1GiB working-set
    margin, host reserve and conservative memory PSI bounds. These margins are
    enforced on every sample, even before a max event, so summaries remain
    directly auditable. OOM, actual swap and unknown readings always fail.
    """
    _validate_sample(value)
    policy = policy_for(policy_version)
    if (value['memory_limit_bytes'] != POLICY['memory_limit_bytes']
            or value['swap_limit_bytes'] != 0
            or not 0 < value['memory_current_bytes'] <= value['memory_limit_bytes']
            or value['inactive_file_bytes'] > value['memory_current_bytes']
            or value['working_set_bytes'] != value['memory_current_bytes']-value['inactive_file_bytes']):
        raise RuntimeError('native v2 fixed cap or working-set accounting differs')
    if (value['host_available_bytes'] < POLICY['minimum_host_available_bytes']
            or value['swap_bytes'] or value['events']['oom'] or value['events']['oom_kill']
            or value['working_set_bytes'] > POLICY['memory_limit_bytes']-POLICY['minimum_working_set_margin_bytes']
            or value['some_avg10'] > policy['maximum_some_avg10']
            or value['full_avg10'] > policy['maximum_full_avg10']):
        raise RuntimeError('owned native host-floor/swap/OOM/working-set/PSI guard failed')
    return dict(limit_pressure_events=value['events']['max'],
                cache_reclaim_events=value['events']['max'], unsafe_pressure_events=0)


def validate_summary(summary):
    """Pure metadata validation; v1 raw max must remain literally zero.

    v2 limit_pressure_events is the ABSOLUTE actual cgroup max counter, never
    reset or relabeled. phase_* counters separately expose its measured delta.
    """
    if (not isinstance(summary, dict) or summary.get('passed') is not True
            or not _integer(summary.get('samples')) or summary['samples'] < 2
            or not _integer(summary.get('minimum_host_available_bytes'))
            or summary['minimum_host_available_bytes'] < 8*GIB
            or any(not _integer(summary.get(k)) for k in ('swap_peak_bytes','oom_events','limit_pressure_events'))
            or summary['swap_peak_bytes'] or summary['oom_events']):
        raise RuntimeError('actual successful native resource summary required')
    if 'counter_read_attempts' in summary or 'counter_read_recoveries' in summary:
        if (not _integer(summary.get('counter_read_attempts'))
                or not _integer(summary.get('counter_read_recoveries'))
                or summary['counter_read_attempts']!=summary['samples']+summary['counter_read_recoveries']
                or summary['counter_read_recoveries']>2*summary['samples']):
            raise RuntimeError('native counter-read recovery audit differs')
    schema = summary.get('schema')
    if schema in (None, 'roborender.native-phase-resource-guard.v1'):
        if summary['limit_pressure_events']:
            raise RuntimeError('legacy v1 pressure cannot be reclassified')
        return summary
    version = summary.get('policy_version')
    policy = policy_for(version)
    sustained = schema == SUSTAINED_SCHEMA and version == SUSTAINED_VERSION
    if sustained:
        pressure = summary.get('pressure_window')
        if (not isinstance(pressure,dict)
                or set(pressure) != {'warning_samples','maximum_warning_seconds','failed'}
                or not _integer(pressure['warning_samples'])
                or pressure['warning_samples'] > summary['samples']
                or type(pressure['maximum_warning_seconds']) not in (int,float)
                or not math.isfinite(pressure['maximum_warning_seconds'])
                or not 0 <= pressure['maximum_warning_seconds'] < policy['maximum_warning_seconds']
                or pressure['failed'] is not False):
            raise RuntimeError('actual bounded v3 pressure history required')
        peaks=(summary.get('maximum_some_avg10'),summary.get('maximum_full_avg10'))
        if (not all(_finite_percent(v) for v in peaks)
                or bool(pressure['warning_samples']) !=
                    (peaks[0]>policy['warning_some_avg10'] or peaks[1]>policy['warning_full_avg10'])
                or (not pressure['warning_samples'] and pressure['maximum_warning_seconds']!=0)):
            raise RuntimeError('v3 pressure peaks and warning history disagree')
    fields = ('cache_reclaim_events','unsafe_pressure_events','initial_limit_pressure_events',
              'phase_limit_pressure_events','phase_cache_reclaim_events','phase_unsafe_pressure_events',
              'memory_limit_bytes','swap_limit_bytes','maximum_working_set_bytes')
    if (not (sustained or (schema == SCHEMA and version == POLICY_VERSION))
            or any(not _integer(summary.get(k)) for k in fields)
            or summary['memory_limit_bytes'] != 21*GIB or summary['swap_limit_bytes'] != 0
            or summary['unsafe_pressure_events'] != 0
            or summary['cache_reclaim_events'] != summary['limit_pressure_events']
            or summary['initial_limit_pressure_events'] > summary['limit_pressure_events']
            or summary['phase_limit_pressure_events'] != summary['limit_pressure_events']-summary['initial_limit_pressure_events']
            or summary['phase_cache_reclaim_events'] != summary['phase_limit_pressure_events']
            or summary['phase_unsafe_pressure_events'] != 0
            or summary['maximum_working_set_bytes'] > 20*GIB
            or not _finite_percent(summary.get('maximum_some_avg10'))
            or not _finite_percent(summary.get('maximum_full_avg10'))
            or summary['maximum_some_avg10'] > policy['maximum_some_avg10']
            or summary['maximum_full_avg10'] > policy['maximum_full_avg10']):
        raise RuntimeError('native v2 resource summary/policy/classification differs')
    return summary


def own_cgroup_root(mount=Path('/sys/fs/cgroup'), membership=Path('/proc/self/cgroup')):
    rows = [line.split(':', 2) for line in Path(membership).read_text().splitlines()]
    unified = [row[2] for row in rows if len(row) == 3 and row[:2] == ['0', '']]
    if len(unified) != 1 or not unified[0].startswith('/') or '..' in Path(unified[0]).parts:
        raise RuntimeError('cannot resolve owned unified cgroup')
    root = Path(mount).resolve(strict=True)
    value = (root / unified[0].lstrip('/')).resolve(strict=True)
    if not value.is_relative_to(root) or not (value / 'memory.events').is_file():
        raise RuntimeError('owned memory cgroup is unavailable')
    return value


class ResourceReadError(RuntimeError):
    """Bounded nonsecret procfs evidence, retained by native and outer guards."""
    def __init__(self, diagnostic):
        self.diagnostic=diagnostic
        last=diagnostic['attempts'][-1]
        issue=(last['fatal_observations'] or last['errors'])[0]
        super().__init__('native resource read failed: '+issue['field']+' ('+issue['reason']+
            '); attempts='+str(len(diagnostic['attempts'])))


def _counter_attempt(root,meminfo,attempt,policy_version=POLICY_VERSION):
    """Collect every file before interpreting errors; never hide later OOM."""
    policy=policy_for(policy_version)
    paths={'meminfo':Path(meminfo)}
    paths.update({name:root/name for name in ('memory.events','memory.pressure','memory.stat',
        'memory.current','memory.max','memory.swap.current','memory.swap.max')})
    raw={};raw_bytes={};errors=[];fatal=[];observed={};result={'events':{}};event_sightings=[]
    def issue(name,field,reason,*,unsafe=False,value=None):
        data=raw_bytes.get(name)
        row=dict(field=field,file=str(paths.get(name,root)),reason=reason,
            raw_excerpt=data.decode('utf8',errors='backslashreplace')[:256] if data is not None else None,
            raw_sha256=hashlib.sha256(data).hexdigest() if data is not None else None)
        if value is not None:row['observed_value']=value
        target=fatal if unsafe else errors
        if len(target)<32:target.append(row)
    try:
        identity=root.stat()
        result.update(process_id=os.getpid(),cgroup_path=str(root),
            cgroup_device=identity.st_dev,cgroup_inode=identity.st_ino)
        validate_identity(result)
    except (OSError,RuntimeError) as exc:
        issue('cgroup','identity','identity unavailable: '+type(exc).__name__,unsafe=True)
    for name,path in paths.items():
        try:
            # These are fixed kernel metadata files, not datasets/models.
            data=path.read_bytes();raw_bytes[name]=data
            if len(data)>65536:raise ValueError('counter file exceeds64KiB')
            raw[name]=data.decode('utf8')
        except (OSError,ValueError,UnicodeError) as exc:
            issue(name,name,'read unavailable: '+type(exc).__name__)
    def emit(name,field,value):
        observed.setdefault(field,[]).append(value)
        if field.startswith('events.'):
            key=field.split('.',1)[1];result['events'][key]=max(result['events'].get(key,value),value);event_sightings.append((key,value))
        else:result[field]=value
        def unsafe(reason):
            issue(name,field,reason,unsafe=True,value=value if not isinstance(value,float) or math.isfinite(value) else None)
        if field in ('some_avg10','full_avg10'):
            if not _finite_percent(value):unsafe('invalid PSI measurement')
            elif value>policy['maximum_'+field]:unsafe('PSI threshold exceeded')
        elif not _integer(value):unsafe('invalid negative counter')
        elif field=='host_available_bytes' and value<8*GIB:unsafe('host reserve below8GiB')
        elif field=='memory_limit_bytes' and value!=21*GIB:unsafe('fixed21GiB cap changed')
        elif field=='swap_limit_bytes' and value!=0:unsafe('zero-swap cap changed')
        elif field=='memory_current_bytes' and not 0<value<=21*GIB:unsafe('current outside fixed cap')
        elif field in ('swap_bytes','events.oom','events.oom_kill') and value:
            unsafe('actual swap/OOM observed')
    def integer(name,field,text,*,factor=1):
        if field in ('memory_limit_bytes','swap_limit_bytes') and text.strip()=='max':
            issue(name,field,'unlimited cap observed',unsafe=True);return
        if not re.fullmatch(r'-?[0-9]+',text.strip()):
            issue(name,field,'malformed integer');return
        try:value=int(text)*factor
        except ValueError:issue(name,field,'malformed integer');return
        emit(name,field,value)
    for name,field in [('memory.current','memory_current_bytes'),('memory.max','memory_limit_bytes'),
            ('memory.swap.current','swap_bytes'),('memory.swap.max','swap_limit_bytes')]:
        if name in raw:integer(name,field,raw[name])
    # Parse rows independently, retaining unsafe values even alongside a
    # malformed/duplicate row elsewhere in the same file.
    for name in ('meminfo','memory.events','memory.stat'):
        if name not in raw:continue
        seen=set()
        for line in raw[name].splitlines():
            if not line.strip():continue
            parts=line.split(':',1) if name=='meminfo' else line.split()
            if len(parts)!=2:issue(name,name,'malformed row');continue
            key,text=parts
            if key in seen:issue(name,key,'duplicate row')
            seen.add(key)
            if name=='meminfo' and key=='MemAvailable':
                fields=text.split()
                if len(fields)!=2 or fields[1]!='kB':issue(name,'host_available_bytes','malformed MemAvailable units')
                else:integer(name,'host_available_bytes',fields[0],factor=1024)
            elif name=='memory.events' and key in ('max','oom','oom_kill'):
                integer(name,'events.'+key,text)
            elif name=='memory.stat' and key=='inactive_file':integer(name,'inactive_file_bytes',text)
        required={'meminfo':('MemAvailable',),'memory.events':('max','oom','oom_kill'),'memory.stat':('inactive_file',)}[name]
        for key in required:
            if key not in seen:issue(name,key,'required counter missing')
    if 'memory.pressure' in raw:
        seen=set()
        for line in raw['memory.pressure'].splitlines():
            fields=line.split()
            if not fields or fields[0] not in ('some','full'):
                issue('memory.pressure','memory.pressure','malformed PSI row');continue
            key=fields[0];found=False
            if key in seen:issue('memory.pressure',key,'duplicate PSI row')
            seen.add(key)
            for token in fields[1:]:
                pair=token.split('=',1)
                if len(pair)!=2:issue('memory.pressure',key,'malformed PSI field');continue
                if pair[0]!='avg10':continue
                if found:issue('memory.pressure',key+'_avg10','duplicate PSI avg10')
                found=True
                try:value=float(pair[1])
                except ValueError:issue('memory.pressure',key+'_avg10','malformed PSI value');continue
                if not re.fullmatch(r'[0-9]+(?:\.[0-9]+)?',pair[1]):
                    issue('memory.pressure',key+'_avg10','malformed PSI numeric format')
                emit('memory.pressure',key+'_avg10',value)
            if not found:issue('memory.pressure',key+'_avg10','PSI avg10 missing')
        for key in ('some','full'):
            if key not in seen:issue('memory.pressure',key+'_avg10','PSI row missing')
    currents=observed.get('memory_current_bytes',[]);inactives=observed.get('inactive_file_bytes',[])
    for current in currents:
        for inactive in inactives:
            if current>=inactive and current-inactive>20*GIB:
                issue('memory.stat','working_set_bytes','working-set margin below1GiB',unsafe=True)
    if currents and inactives:
        if currents[-1]<inactives[-1]:issue('memory.stat','working_set_bytes','inconsistent current/inactive_file snapshot')
        else:result['working_set_bytes']=currents[-1]-inactives[-1]
    try:
        after=root.stat()
        if (after.st_dev,after.st_ino)!=(result.get('cgroup_device'),result.get('cgroup_inode')):
            issue('cgroup','identity','cgroup identity changed during read',unsafe=True)
    except OSError:issue('cgroup','identity','cgroup identity disappeared during read',unsafe=True)
    # JSON receipts cannot contain NaN/Inf; the raw excerpt preserves that
    # observation, while the parsed invalid measurement is explicitly null.
    safe_result={k:v for k,v in result.items()}
    for key in ('some_avg10','full_avg10'):
        if key in safe_result and not math.isfinite(safe_result[key]):safe_result[key]=None
    return result,dict(attempt=attempt,errors=errors,fatal_observations=fatal,observed_sample=safe_result),event_sightings


def sample_resources(*, cgroup_root=None, meminfo=Path('/proc/meminfo'),policy_version=POLICY_VERSION):
    policy_for(policy_version)
    try:root=own_cgroup_root() if cgroup_root is None else Path(cgroup_root)
    except (OSError,RuntimeError) as exc:
        raise ResourceReadError(dict(schema='roborender.native-resource-read-error.v1',maximum_attempts=3,
            attempts=[dict(attempt=1,errors=[],fatal_observations=[dict(field='identity',
                file='/proc/self/cgroup',reason='owned cgroup resolution failed: '+type(exc).__name__)],
                observed_sample={})])) from exc
    diagnostics=[];identity=None;counters={}
    for attempt in range(1,4):
        result,diagnostic,sightings=_counter_attempt(root,meminfo,attempt,policy_version)
        if 'cgroup_inode' in result:
            current=validate_identity(result)
            if identity is not None and current!=identity:
                diagnostic['fatal_observations'].append(dict(field='identity',file=str(root),reason='cgroup identity changed between read attempts'))
            identity=current
        for key,value in sightings:
            if key in counters and value<counters[key]:
                diagnostic['fatal_observations'].append(dict(field='events.'+key,file=str(root/'memory.events'),reason='event counter regressed between reads'))
            counters[key]=value
        if diagnostic['fatal_observations']:
            diagnostics.append(diagnostic)
            raise ResourceReadError(dict(schema='roborender.native-resource-read-error.v1',maximum_attempts=3,attempts=diagnostics))
        if not diagnostic['errors']:
            _validate_sample(result)
            result.update(read_attempts=attempt,recovered_read_errors=diagnostics)
            return result
        diagnostics.append(diagnostic)
    raise ResourceReadError(dict(schema='roborender.native-resource-read-error.v1',maximum_attempts=3,attempts=diagnostics))


class PhaseResourceGuard:
    def __init__(self, output, *, sample=None, interval_s=.5, policy_version=POLICY_VERSION,
                 clock=time.monotonic):
        if not 0 < interval_s <= .5:
            raise ValueError('native memory monitor interval must be <=0.5 seconds')
        self.path = Path(output)
        if self.path.exists():
            raise RuntimeError('a native resource phase cannot be replayed')
        self.policy_version = policy_version
        self.policy = policy_for(policy_version)
        self.schema = SUSTAINED_SCHEMA if policy_version == SUSTAINED_VERSION else SCHEMA
        self.sample = sample or (lambda: sample_resources(policy_version=policy_version))
        self.pressure = PressureWindow() if policy_version == SUSTAINED_VERSION else None
        self.clock = clock
        self.interval_s = interval_s
        self.stop = threading.Event()
        self.lock = threading.Lock()
        self.samples = 0
        self.first = self.last = None
        self.minimum = None
        self.peak = self.swap = 0
        self.working_set_peak = 0
        self.some_peak = self.full_peak = 0.0
        self.unsafe_memory = False
        self.error = None
        self.sampling_diagnostic = None
        self.failed_sample = None
        self.counter_read_attempts = self.counter_read_recoveries = 0
        self.recovered_read_diagnostics = []
        self.read_diagnostics_truncated = False
        self.started_ns = None

    def _sample(self, *, final=False):
        # Never hold the summary lock across procfs I/O. A stuck sampling
        # callback must not turn a bounded join into an unbounded finalizer.
        try:
            value = self.sample()
            _validate_sample(value)
            with self.lock:
                if self.stop.is_set() and not final:
                    return
                self.samples += 1
                attempts=value.get('read_attempts',1)
                if type(attempts) is not int or not 1<=attempts<=3:
                    raise RuntimeError('bounded actual counter-read attempts required')
                self.counter_read_attempts+=attempts
                self.counter_read_recoveries+=attempts-1
                self.recovered_read_diagnostics.extend(value.get('recovered_read_errors',[]))
                if len(self.recovered_read_diagnostics)>16:
                    self.read_diagnostics_truncated=True
                    self.recovered_read_diagnostics=self.recovered_read_diagnostics[-16:]
                self.first = self.first or value
                self.last = value
                previous = getattr(self, '_previous', None)
                try:
                    validate_transition(previous,value)
                except RuntimeError as exc:
                    self.error = self.error or str(exc)
                self._previous = value
                self.minimum = min(self.minimum, value['host_available_bytes']) if self.minimum is not None else value['host_available_bytes']
                self.peak = max(self.peak, value['memory_current_bytes'])
                self.swap = max(self.swap, value['swap_bytes'])
                self.working_set_peak = max(self.working_set_peak, value['working_set_bytes'])
                self.some_peak = max(self.some_peak, value['some_avg10'])
                self.full_peak = max(self.full_peak, value['full_avg10'])
                try:
                    classify_sample(value,policy_version=self.policy_version)
                    if self.pressure is not None: self.pressure.observe(value,self.clock())
                except RuntimeError as exc:
                    self.unsafe_memory = True
                    self.error = self.error or str(exc)
        except Exception as exc:
            with self.lock:
                if self.stop.is_set() and not final:
                    return
                if isinstance(exc,ResourceReadError):
                    self.sampling_diagnostic=self.sampling_diagnostic or exc.diagnostic
                    self.failed_sample=self.failed_sample or exc.diagnostic['attempts'][-1]['observed_sample']
                    self.unsafe_memory=self.unsafe_memory or any(x['fatal_observations'] for x in exc.diagnostic['attempts'])
                    self.counter_read_attempts+=len(exc.diagnostic['attempts'])
                    self.error=self.error or str(exc)
                else:self.error = self.error or f'resource sampling failed: {type(exc).__name__}'

    def _run(self):
        while not self.stop.wait(self.interval_s):
            self._sample()

    def check(self):
        if self.error:
            raise RuntimeError(self.error)

    def snapshot(self, output):
        """Seal actual prefix counters without stopping the live monitor.

        The output is distinct from the terminal resource receipt.  Counters,
        pressure history and native identity are never reset at this barrier.
        A prefix receipt is not evidence for the as-yet unfinished round.
        """
        output = Path(output)
        if output == self.path or output.exists():
            raise RuntimeError('resource prefix snapshot requires a fresh distinct path')
        self._sample()
        with self.lock:
            self.check()
            self._publish(None, output=output)
            value = dict(self.summary, scope='continuous_round_prefix')
            validate_summary(value)
            return value

    def __enter__(self):
        self.started_ns = time.time_ns()
        self._sample()
        if self.error:
            self._publish(RuntimeError)
        self.check()
        self.thread = threading.Thread(target=self._run, name='native-phase-resource-guard', daemon=True)
        self.thread.start()
        return self

    def __exit__(self, kind, value, trace):
        self.stop.set()
        self.thread.join(timeout=2)
        if self.thread.is_alive():
            self.error = self.error or 'resource monitor failed to stop'
        else:
            self._sample(final=True)
        self._publish(kind)
        if kind is None:
            self.check()
        return False

    def _publish(self, kind, *, output=None):
        # Keep policy/summary readers stdlib-only on the host; persistence's
        # contracts dependency is needed only inside the native phase owner.
        from .durable import atomic_write_json
        if self.first is None or self.last is None or self.samples < 2:
            self.error = self.error or 'resource monitor lacks complete actual samples'
        final=self.failed_sample if self.failed_sample is not None else self.last
        final_events=final.get('events',{}) if final is not None else {}
        raw_max = final_events.get('max')
        initial_max = self.first['events']['max'] if self.first else None
        phase_max = max(0, raw_max-initial_max) if raw_max is not None and initial_max is not None else None
        classification_unknown=self.failed_sample is not None and not self.unsafe_memory
        cache = ((raw_max if not self.unsafe_memory else 0)
            if raw_max is not None and not classification_unknown else None)
        self.receipt = dict(schema=self.schema, policy_version=self.policy_version, policy=dict(self.policy),
            started_ns=self.started_ns, finished_ns=time.time_ns(), samples=self.samples,
            interval_s=self.interval_s, minimum_host_available_bytes=self.minimum,
            memory_current_peak_bytes=self.peak if self.samples else None,
            swap_peak_bytes=(max(self.swap,final['swap_bytes']) if final is not None and _integer(final.get('swap_bytes')) else None),
            oom_events=(sum(final_events[key] for key in ('oom','oom_kill'))
                if all(_integer(final_events.get(key)) for key in ('oom','oom_kill')) else None),
            limit_pressure_events=raw_max, cache_reclaim_events=cache,
            unsafe_pressure_events=(raw_max-cache if raw_max is not None and cache is not None else None),
            initial_limit_pressure_events=initial_max, phase_limit_pressure_events=phase_max,
            phase_cache_reclaim_events=(phase_max if not self.unsafe_memory else 0) if phase_max is not None and not classification_unknown else None,
            phase_unsafe_pressure_events=(phase_max if self.unsafe_memory else 0) if phase_max is not None and not classification_unknown else None,
            memory_limit_bytes=final.get('memory_limit_bytes') if final is not None else None,
            swap_limit_bytes=final.get('swap_limit_bytes') if final is not None else None,
            maximum_working_set_bytes=self.working_set_peak if self.samples else None,
            maximum_some_avg10=self.some_peak if self.samples else None,
            maximum_full_avg10=self.full_peak if self.samples else None,
            first_sample=self.first, last_sample=self.last, failed_sample=self.failed_sample,sampling_error=self.error,
            sampling_diagnostic=self.sampling_diagnostic,
            counter_read_attempts=self.counter_read_attempts,counter_read_recoveries=self.counter_read_recoveries,
            recovered_read_diagnostics=self.recovered_read_diagnostics,
            read_diagnostics_truncated=self.read_diagnostics_truncated,
            passed=kind is None and self.error is None)
        if self.pressure is not None: self.receipt['pressure_window'] = self.pressure.summary()
        # This contains resource counters, not credentials. Native containers
        # run as root while the host-side qualification reader runs as ldu.
        path = self.path if output is None else Path(output)
        atomic_write_json(path, self.receipt, mode=0o644)
        raw = path.read_bytes()
        self.summary = {key: self.receipt[key] for key in SUMMARY_FIELDS}
        if self.pressure is not None: self.summary['pressure_window'] = self.receipt['pressure_window']
        self.summary.update(path=str(path), raw_sha256=hashlib.sha256(raw).hexdigest(),
            samples=self.samples, passed=self.receipt['passed'],counter_read_attempts=self.counter_read_attempts,
            counter_read_recoveries=self.counter_read_recoveries)
