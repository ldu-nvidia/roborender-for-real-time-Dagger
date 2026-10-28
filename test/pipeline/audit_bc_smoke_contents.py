"""Read-only deep audit of completed diagnostic BC chunks; no dataset admission.

Decompress one array at a time, including all camera arrays. Camera raw depth
may legitimately contain infinity (no hit); report it rather than rewriting it.
Run only after the paired terminal PASS and the separate hash/resource audit.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys


def inspect_array(name, value):
    import numpy as np

    if value.dtype.hasobject:
        raise ValueError('object arrays are forbidden')
    if name in ('env_id', 'seq', 'prompt'):
        return dict(shape=list(value.shape), dtype=str(value.dtype))
    if value.shape[0] != 81:
        raise ValueError('every payload stream must contain81 rows')
    floating = np.issubdtype(value.dtype, np.floating)
    finite = np.isfinite(value) if floating else None
    nonfinite = int(value.size - np.count_nonzero(finite)) if floating else 0
    if (name == 'gt_actions' or name.startswith('obs_')) and nonfinite:
        raise ValueError('nonfinite teacher action or observation')
    if name.startswith('depth_rawq__') and int(value.max()) > 16383:
        raise ValueError('quantized depth exceeds Raven14-bit range')
    if name == 'dones' and value.any():
        raise ValueError('native serialized dones convention changed')
    valid = value[finite] if floating else value
    return dict(shape=list(value.shape), dtype=str(value.dtype), elements=int(value.size),
                nonfinite=nonfinite, nonzero=int(np.count_nonzero(value)),
                finite_min=float(valid.min()) if valid.size else None,
                finite_max=float(valid.max()) if valid.size else None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', type=Path, required=True)
    parser.add_argument('--source-root', type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.source_root))
    import numpy as np
    from pipeline.bc_collection.continuous_ingest import _validate_row, validate_payload

    read = lambda path: json.loads(path.read_bytes())
    result = read(args.stage / 'pair-result.json')
    if result.get('verdict') != 'PASS' or result.get('failure') is not None:
        raise ValueError('terminal paired smoke PASS required before content audit')
    summaries = []
    for actor, binding in enumerate(result['actors']):
        root = Path(binding['runtime']) / 'bc_isaac'
        request = read(args.stage / f'request{actor}.json')
        emitted = [0] * len(request['motions'])
        previous, arrays, episodes = {}, {}, Counter()
        terminal_rows = cross_window_chunks = count = 0
        index = root / 'chunks.jsonl'
        with index.open() as stream:
            for ordinal, line in enumerate(stream):
                record = json.loads(line)
                payload = root / Path(record['source_path']).name
                sidecar = root / Path(record['provenance_path']).name
                provenance = read(sidecar)
                _validate_row(record, provenance, request, ordinal=ordinal, emitted=emitted,
                              previous=previous, window=record['emitted_at_window'])
                validate_payload(payload, record)
                with np.load(payload, allow_pickle=False) as archive:
                    for name in archive.files:
                        stats = inspect_array(name, archive[name])
                        if 'elements' not in stats:
                            continue
                        combined = arrays.setdefault(name, dict(chunks=0, elements=0,
                            nonfinite=0, nonzero=0, finite_min=None, finite_max=None,
                            shape=stats['shape'], dtype=stats['dtype']))
                        if (combined['shape'], combined['dtype']) != (stats['shape'], stats['dtype']):
                            raise ValueError('array schema changed across chunks')
                        combined['chunks'] += 1
                        for key in ('elements', 'nonfinite', 'nonzero'):
                            combined[key] += stats[key]
                        for key, fn in (('finite_min', min), ('finite_max', max)):
                            value = stats[key]
                            if value is not None:
                                combined[key] = value if combined[key] is None else fn(combined[key], value)
                rows = provenance['rows']
                terminal_rows += sum(row['actual_done'] for row in rows)
                cross_window_chunks += int(rows[0]['vector_step_index'] // 81 != rows[-1]['vector_step_index'] // 81)
                episodes[provenance['episode_id']] += 1
                count += 1
        summaries.append(dict(actor=actor, chunks=count, per_lane=emitted,
            represented_episodes=len(episodes), terminal_rows_in_saved_chunks=terminal_rows,
            chunks_crossing_capture_window=cross_window_chunks, arrays=arrays,
            index_sha256=hashlib.sha256(index.read_bytes()).hexdigest()))
    print(json.dumps(dict(verdict='PASS', scope='completed-smoke-content-audit',
        production_admitted=False, all_arrays_decompressed=True,
        chronological_episode_rows_verified=True, actors=summaries)))


if __name__ == '__main__':
    main()
