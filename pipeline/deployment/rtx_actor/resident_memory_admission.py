"""Conservative cap-change admission; no cache credit or runtime guarantee.

MemAvailable already credits reclaimable file cache. Adding memory.current
double-credits that cache. Only measured resident anonymous/shmem bytes are
credited against a future cgroup cap here; kernel and all file cache are
deliberately not credited. MemAvailable is still an OS estimate, not an OOM
guarantee: runtime headroom/swap/OOM monitoring remains mandatory.
"""
GIB = 1024**3


def admit_future_caps(mem_available_bytes, owners, target_caps_bytes, *, floor_bytes=8*GIB):
    if (type(mem_available_bytes) is not int or mem_available_bytes < 0
            or type(floor_bytes) is not int or floor_bytes < 8*GIB
            or not owners or len(owners) != len(target_caps_bytes)):
        raise ValueError('explicit complete host/owner attribution and >=8GiB floor required')
    rows = []
    names = set()
    for owner, cap in zip(owners, target_caps_bytes, strict=True):
        name = owner.get('container')
        current = owner.get('memory_current')
        stat = owner.get('stat', {})
        anon, shmem, kernel = (stat.get(key) for key in ('anon', 'shmem', 'kernel'))
        events = owner.get('events', {})
        if (not isinstance(name, str) or not name or name in names
                or any(type(value) is not int or value < 0 for value in (current, anon, shmem, kernel))
                or type(cap) is not int or cap <= 0
                or owner.get('memory_swap.current') != 0
                or owner.get('sampling_error') or owner.get('oom_killed')
                or not isinstance(events, dict)
                or not {'max', 'oom', 'oom_kill'} <= set(events)
                or any(type(events[key]) is not int or events[key] != 0
                       for key in ('max', 'oom', 'oom_kill'))):
            raise ValueError('missing, unhealthy, or duplicate owner attribution')
        names.add(name)
        credit = anon + shmem
        if credit > current:
            raise ValueError('inconsistent/non-atomic cgroup counters; resample, never invent credit')
        if credit + kernel + GIB > cap:
            raise ValueError('target cap lacks 1GiB margin over currently committed private/kernel memory')
        rows.append({'container': name, 'current_anon_shmem_bytes': credit,
                     'target_cap_bytes': cap, 'additional_budget_bytes': max(0, cap-credit)})
    additional = sum(row['additional_budget_bytes'] for row in rows)
    projected_floor = mem_available_bytes-additional
    return {'schema': 'roborender.resident-memory-admission.v1',
            'admitted': projected_floor >= floor_bytes,
            'mem_available_bytes': mem_available_bytes,
            'future_additional_budget_bytes': additional,
            'projected_headroom_bytes': projected_floor,
            'required_headroom_bytes': floor_bytes,
            'file_cache_credit_bytes': 0, 'owners': rows,
            'runtime_monitoring_required': True}
