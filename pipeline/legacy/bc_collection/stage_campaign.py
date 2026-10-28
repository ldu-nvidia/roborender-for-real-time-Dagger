"""Build launch metadata after actual-B capacity receipts exist; never execute."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from pipeline.legacy.bc_collection.campaign_controller import SCHEMA, STORAGE_SCHEMA, binding, bound, publish, read_raw, validate_campaign, validate_storage_policy


def build_campaign(schedule, *, campaign_id, source_root, source_manifest,
                   capacity_receipts, timeout_seconds=7200, first_session_timeout_seconds=None,
                   storage_root=None, storage_reserve_bytes=64 * 2**30,
                   storage_bytes_per_emission_budget=512 * 2**20,
                   storage_session_metadata_budget_bytes=2**30,
                   runtime_prefix='/home/ldu/roborender/runtime/bc-sessions'):
    for value in (timeout_seconds, first_session_timeout_seconds):
        if value is not None and (type(value) is not int or not 60 <= value <= 12 * 3600):
            raise ValueError('session timeouts must be integer seconds between 60 and 43200')
    if timeout_seconds is None:
        raise ValueError('default session timeout is required')
    if (schedule.get('schema') != 'roborender.bc-native-schedule.v1'
            or not schedule.get('sessions') or any(row.get('physical_gpu') != 0 for row in schedule['sessions'])):
        raise ValueError('explicit serial GPU0 schedule required')
    capacities = {}
    for path in capacity_receipts:
        ref = binding(path); receipt = bound(ref)
        size = receipt.get('num_envs')
        if size in capacities:
            raise ValueError('duplicate actual-B capacity receipt')
        capacities[size] = (ref, receipt)
    expected = {row['num_envs'] for row in schedule['sessions']}
    if set(capacities) != expected:
        raise ValueError('every actual B and tail requires exactly one capacity receipt')
    caps = {receipt['cgroup_limit_bytes'] for _, receipt in capacities.values()}
    if len(caps) != 1 or next(iter(caps)) not in (40 * 2**30, 43 * 2**30):
        raise ValueError('one predeclared qualified memory cap is required')
    rows = []
    for index, row in enumerate(schedule['sessions']):
        session_ref = dict(path=row['session_path'], raw_sha256=row['session_sha256'])
        session = bound(session_ref)
        if row['session_id'] != session['session_id'] or len(session['requests']) != 1:
            raise ValueError('single-cohort capacity evidence requires --cohorts-per-session 1; multi-scene lifetime is unqualified')
        rows.append(dict(session_id=row['session_id'], session=session_ref, requests_dir=row['requests_dir'],
            runtime=str(Path(runtime_prefix) / row['session_id']),
            capacity_receipt=capacities[row['num_envs']][0],
            timeout_seconds=(first_session_timeout_seconds if index == 0 and first_session_timeout_seconds is not None
                             else timeout_seconds)))
    storage = validate_storage_policy(dict(schema=STORAGE_SCHEMA,
        root=str(Path(storage_root) if storage_root is not None else Path(runtime_prefix).parent),
        reserve_bytes=storage_reserve_bytes, bytes_per_emission_budget=storage_bytes_per_emission_budget,
        session_metadata_budget_bytes=storage_session_metadata_budget_bytes, poll_seconds=1))
    return dict(schema=SCHEMA, campaign_id=campaign_id, source_root=str(Path(source_root).resolve(strict=True)),
        source_manifest=binding(source_manifest), physical_gpu=0,
        memory_gib=next(iter(caps)) // 2**30, sessions=rows, storage_guard=storage)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--schedule', type=Path, required=True)
    parser.add_argument('--schedule-sha256', required=True)
    parser.add_argument('--campaign-id', required=True)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--source-manifest', type=Path, required=True)
    parser.add_argument('--capacity-receipts', type=Path, nargs='+', required=True)
    parser.add_argument('--timeout-seconds', type=int, default=7200)
    parser.add_argument('--first-session-timeout-seconds', type=int,
                        help='Optional bounded first-session override; later sessions keep --timeout-seconds')
    parser.add_argument('--storage-root', type=Path,
                        help='Existing runtime filesystem ancestor; default /home/ldu/roborender/runtime')
    parser.add_argument('--storage-reserve-gib', type=int, default=64)
    parser.add_argument('--storage-chunk-budget-mib', type=int, default=512,
                        help='Conservative declared budget, not an observed compression guarantee; minimum512MiB')
    parser.add_argument('--storage-session-metadata-budget-gib', type=int, default=1)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    schedule = bound(dict(path=str(args.schedule.resolve(strict=True)), raw_sha256=args.schedule_sha256))
    doc = build_campaign(schedule, campaign_id=args.campaign_id, source_root=args.source_root,
        source_manifest=args.source_manifest, capacity_receipts=args.capacity_receipts,
        timeout_seconds=args.timeout_seconds, first_session_timeout_seconds=args.first_session_timeout_seconds,
        storage_root=args.storage_root, storage_reserve_bytes=args.storage_reserve_gib * 2**30,
        storage_bytes_per_emission_budget=args.storage_chunk_budget_mib * 2**20,
        storage_session_metadata_budget_bytes=args.storage_session_metadata_budget_gib * 2**30)
    validate_campaign(doc)
    publish(args.output, doc)
    print(json.dumps(dict(campaign=binding(args.output), sessions=len(doc['sessions']), execute=False,
                          production_admitted=False), sort_keys=True))


if __name__ == '__main__':
    main()
