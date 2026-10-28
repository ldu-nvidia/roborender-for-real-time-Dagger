"""Compact evidence from committed native done/timeout outcomes and lane keys.

This is not a simulator snapshot or a replay capability. It proves the actual
actor cursor sequence at rollout boundaries; native reset receipts separately
attest robot/object initialization. Model buffer clearing is not an episode reset.
"""
from __future__ import annotations


def cursor_rows(keys):
    return [dict(lane=i, episode_id=k.episode_id, episode_step_index=k.episode_step_index)
            for i, k in enumerate(keys)]


def validate_window(value, *, batch, index, previous_end=None):
    if (not isinstance(value, dict) or set(value) != {'schema','window_index','start','end','native_resets'}
            or value['schema'] != 'roborender.native-window-cursor-continuity.v1'
            or type(value['window_index']) is not int or value['window_index'] != index):
        raise ValueError('exact native window continuity evidence required')
    start, end, resets = value['start'], value['end'], value['native_resets']
    for rows in (start, end):
        if not isinstance(rows,list) or len(rows) != batch:
            raise ValueError('native continuity requires every lane')
        for lane, row in enumerate(rows):
            if (set(row) != {'lane','episode_id','episode_step_index'} or type(row['lane']) is not int
                    or row['lane'] != lane or not isinstance(row['episode_id'], str)
                    or type(row['episode_step_index']) is not int or row['episode_step_index'] < 0):
                raise ValueError('native lane cursor malformed')
    if previous_end is not None and start != previous_end:
        raise ValueError('native episode cursor changed at a window boundary')
    if not isinstance(resets,list):
        raise ValueError('actual native reset outcomes required')
    expected = [dict(row) for row in start]
    events = {}
    for event in resets:
        if (set(event) != {'lane','step','done','timeout'}
                or type(event['lane']) is not int or not 0 <= event['lane'] < batch
                or type(event['step']) is not int or not 0 <= event['step'] < 81
                or type(event['done']) is not bool or type(event['timeout']) is not bool
                or not (event['done'] or event['timeout'])):
            raise ValueError('episode reset lacks actual done/timeout reason')
        key = (event['step'], event['lane'])
        if key in events:raise ValueError('duplicate native reset event')
        events[key] = event
    for step in range(81):
        for lane,row in enumerate(expected):
            if (step,lane) in events:
                prefix,serial = row['episode_id'].rsplit('-',1)
                if not serial.isdigit():raise ValueError('native episode serial malformed')
                row['episode_id'] = f'{prefix}-{int(serial)+1:08d}'
                row['episode_step_index'] = 0
            else:
                row['episode_step_index'] += 1
    if end != expected:
        raise ValueError('native cursor advance is not explained by committed done/timeout outcomes')
    return value
