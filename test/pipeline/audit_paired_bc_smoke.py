"""Read-only, post-completion payload and memory audit for the B32x2 smoke."""
import argparse
import collections
import hashlib
import json
from pathlib import Path
import sys
import zipfile


def read(path):
    return json.loads(path.read_bytes())


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', type=Path, required=True)
    parser.add_argument('--source-root', type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.source_root))
    import numpy as np
    from pipeline.bc_collection.run_continuous import validate_completion, verify_source
    from pipeline.deployment.rtx_actor.paired_bc_guard import validate_config

    stage = args.stage
    result = read(stage / 'pair-result.json')
    assert result['verdict'] == 'PASS' and result['failure'] is None
    assert result['production_admitted'] is False and result['retain_compute'] is True
    source = read(args.source_root.parent / 'source-manifest.json')
    assert source['source_sha256'] == result['source_sha256']
    verify_source(args.source_root, source)
    config = validate_config(read(stage / 'config.json'))
    assert config['source_sha256'] == source['source_sha256']
    summaries, saves, captures = [], [], [[], []]
    for actor in range(2):
        root = Path(result['actors'][actor]['runtime'])
        request_path = stage / f'request{actor}.json'
        request = read(request_path)
        assert sha(request_path) == config['request_sha256'][actor]
        native = read(root / 'bc_isaac/cohort-manifest.json')
        validate_completion(native, request, sha(request_path))
        assert native['initial_reset_all_calls'] == 1 and native['optimizer_steps'] == 0
        assert native['accepted_windows'] == 4 and native['vector_steps'] == 324
        index = root / 'bc_isaac/chunks.jsonl'
        assert sha(index) == native['chunk_index']['sha256']
        assert index.stat().st_size == native['chunk_index']['bytes']
        rows = [json.loads(line) for line in index.read_bytes().splitlines()]
        counts, seen = collections.Counter(), set()
        for row in rows:
            assert row['frames'] == 81 and row['teacher_only'] and row['selected_for_split']
            assert not row['overcollection'] and row['selection_policy'] == 'all_complete_emissions_v1'
            for prefix in ('source', 'provenance'):
                logical = Path(row[prefix + '_path'])
                assert logical.parent == Path('/dagger-runtime/bc_isaac')
                path = root / 'bc_isaac' / logical.name
                assert not path.is_symlink() and path.stat().st_size == row[prefix + '_bytes']
                assert sha(path) == row[prefix + '_sha256']
            path = root / 'bc_isaac' / Path(row['source_path']).name
            assert path.name not in seen
            seen.add(path.name)
            counts[row['local_env_id']] += 1
            with zipfile.ZipFile(path) as archive:
                for member in archive.namelist():
                    if member in ('prompt.npy', 'env_id.npy', 'seq.npy'):
                        continue
                    with archive.open(member) as stream:
                        version = np.lib.format.read_magic(stream)
                        shape, _, dtype = (np.lib.format.read_array_header_1_0(stream)
                            if version == (1, 0) else np.lib.format.read_array_header_2_0(stream))
                        assert not dtype.hasobject and shape[0] == 81, (path.name, member, shape)
        assert len(rows) == native['emitted_chunks'] == native['chunk_index']['count']
        assert [counts[lane] for lane in range(32)] == native['emitted_per_lane']
        assert set(p.name for p in (root / 'bc_isaac').glob('*.npz')) == seen
        events = [json.loads(line) for line in (stage / f'actor{actor}-events.jsonl').read_text().splitlines()]
        active_save = active_capture = None
        for event in events:
            assert event['phase'] != 'memory_rejected'
            if event['phase'] == 'save_begin':
                assert active_save is None
                active_save = event['time']
            elif event['phase'] == 'save_end':
                assert active_save is not None
                saves.append((active_save, event['time'], actor))
                active_save = None
            elif event['phase'] == 'capture_begin':
                assert active_capture is None
                active_capture = event['time']
            elif event['phase'] == 'capture_end':
                assert active_capture is not None
                captures[actor].append((active_capture, event['time']))
                active_capture = None
        assert active_save is None and active_capture is None and len(captures[actor]) == 4
        summaries.append(dict(actor=actor, lanes=32, windows=4, vector_steps=324,
            chunks=len(rows), per_lane=[counts[lane] for lane in range(32)],
            payload_bytes=sum(row['source_bytes'] for row in rows), native_manifest_sha256=sha(root / 'bc_isaac/cohort-manifest.json')))
    assert len(saves) == 8
    saves.sort()
    assert all(a[1] <= b[0] for a, b in zip(saves, saves[1:]))
    assert any(max(a[0], b[0]) < min(a[1], b[1]) for a in captures[0] for b in captures[1])
    samples = [json.loads(line) for line in (stage / 'memory-samples.jsonl').read_text().splitlines()]
    for row in samples:
        assert 'failure' not in row and row['host']['MemAvailable'] * 1024 >= 4 * 2**30
        for actor in row['actors']:
            container = actor['container']
            if container.get('pid'):
                assert container['memory_swap.current'] == 0 and container['oom_killed'] is False
                assert all(container['events'][key] == 0 for key in ('max', 'oom', 'oom_kill'))
    print(json.dumps(dict(verdict='PASS', production_admitted=False, actors=summaries,
        payload_and_provenance_hashes_verified=True, all_array_frame_headers_verified=True,
        parallel_capture_intervals=True, exclusive_saves_verified=True,
        minimum_host_available_bytes=min(row['host']['MemAvailable'] * 1024 for row in samples),
        peak_simultaneous_cgroup_current_bytes=max(sum(a['container'].get('memory_current', 0)
            for a in row['actors']) for row in samples),
        source_sha256=source['source_sha256'], pair_result_sha256=sha(stage / 'pair-result.json'))))


if __name__ == '__main__':
    main()
