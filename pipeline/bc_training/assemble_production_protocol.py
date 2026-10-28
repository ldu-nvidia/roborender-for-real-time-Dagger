"""Assemble a production protocol from completed native and transfer evidence.

This management tool never creates native, capacity, renderer, or success
receipts. It reads small metadata and destination sizes, not dataset tensors.
The ordinary strict protocol validator remains the independent final gate.
"""
from __future__ import annotations
import argparse
from collections import Counter
import json
from pathlib import Path

from pipeline.bc_training.handoff_common import binding, bound, canonical, cli_binding, digest, publish, read
from pipeline.legacy.bc_collection.ingest_native_cohorts import build_pool
from pipeline.bc_training import production_protocol as strict

ASSETS = 'pipeline/dagger/rtx/config/assets.manifest.json'
ADAPTER = 'pipeline/deployment/rtx_actor/isaac6_bc_production.py'
FILES = {'launch.json', 'host-result.json', 'bc-session-result.json',
         'session-execution.json', 'execution-receipt.json'}
RENDER_KEYS = {'renderer_registry', 'renderer_source_manifest', 'render_asset_identity', 'render_model_sha256'}

def source_expectations(source_root, manifest_binding):
    manifest = bound(manifest_binding)
    rows = manifest['files']
    if digest(canonical(rows)) != manifest['source_sha256'] or len({r['path'] for r in rows}) != len(rows):
        raise ValueError('invalid expected native source manifest')
    by_path = {r['path']: r for r in rows}
    asset_path = Path(source_root) / ASSETS
    raw = read(asset_path)
    if digest(raw) != by_path[ASSETS]['sha256'] or len(raw) != by_path[ASSETS]['bytes']:
        raise ValueError('expected native model manifest differs from frozen source')
    names = {'privileged_teacher': 'teacher', 'homie_walk': 'homie_walk', 'homie_stand': 'homie_stand'}
    assets = json.loads(raw)['artifacts']
    models = {names[r['role']]: r['sha256'] for r in assets if r['role'] in names}
    if len([r for r in assets if r['role'] in names]) != 3 or set(models) != strict.MODEL_NAMES:
        raise ValueError('exact teacher/Homie model pins required')
    return manifest, models, by_path[ADAPTER]['sha256']

def execution_chain(root, cohort_binding, expected_source, expected_source_binding, expected_adapter):
    cohort = bound(cohort_binding)
    request = cohort['request']
    cohort_id = request['cohort_id']
    audit = root / 'manifests/cohorts' / cohort_id
    handoff = bound(binding(audit / 'execution-transfer-receipt.json'))
    if (handoff.get('schema') != 'roborender.bc-execution-metadata-transfer.v1'
            or handoff.get('verified') is not True or handoff.get('cohort_id') != cohort_id
            or handoff.get('cohort_manifest_raw_sha256') != cohort_binding['raw_sha256']
            or set(handoff.get('files', {})) != FILES):
        raise ValueError('matching five-file execution transfer is required')
    bindings, docs = {}, {}
    for name, row in handoff['files'].items():
        path = Path(row['path'])
        expected = (audit / name if name == 'execution-receipt.json'
                    else root / 'manifests/sessions' / handoff['session_id'] / name)
        if path != expected or not path.resolve().is_relative_to(root) or row.get('bytes') != path.stat().st_size:
            raise ValueError('execution transfer path/size differs from scoped metadata')
        bindings[name] = {k: row[k] for k in ('path', 'raw_sha256')}
        docs[name] = bound(bindings[name])
    launch, host, native, session, receipt = [docs[n] for n in
        ('launch.json', 'host-result.json', 'bc-session-result.json', 'session-execution.json', 'execution-receipt.json')]
    execution = session.get('collector_execution', {})
    if (session.get('schema') != 'roborender.bc-session-execution.v1' or session.get('verdict') != 'PASS'
            or session.get('session_id') != handoff['session_id']
            or session.get('native_initializations') != 1
            or session.get('native_result_raw_sha256') != bindings['bc-session-result.json']['raw_sha256']
            or session.get('host_result_raw_sha256') != bindings['host-result.json']['raw_sha256']
            or session.get('launch_raw_sha256') != bindings['launch.json']['raw_sha256']
            or session.get('source_manifest_raw_sha256') != expected_source_binding['raw_sha256']
            or execution != launch.get('collector_execution')
            or execution.get('source_release_sha256') != expected_source['source_sha256']
            or execution.get('adapter_source_sha256') != expected_adapter
            or launch.get('source_manifest_raw_sha256') != expected_source_binding['raw_sha256']
            or launch.get('source_sha256') != expected_source['source_sha256']
            or launch.get('qualification_only') is not False
            or launch.get('session_sha256') != session.get('session_sha256')
            or launch.get('session_canonical_sha256') != session.get('session_canonical_sha256')
            or launch.get('port_manifest_raw_sha256') != session.get('port_manifest_raw_sha256')
            or launch.get('request_identities') != session.get('request_identities')
            or host.get('passed') is not True or host.get('exit_code') != 0 or host.get('failure') is not None
            or native.get('status') != 'complete' or native.get('failure') is not None
            or native.get('session_id') != session['session_id'] or native.get('native_initializations') != 1
            or native.get('scene_initializations') != session['scene_initializations']
            or len(native.get('cohorts', [])) != session['scene_initializations']
            or len(session.get('request_identities', {})) != session['scene_initializations']):
        raise ValueError('native session/host/source evidence is incomplete or changed')
    req_sha, req_canonical = cohort['request_sha256'], digest(canonical(request))
    matches = [row for row in native['cohorts'] if row.get('cohort_id') == cohort_id]
    if (len(matches) != 1 or matches[0].get('request_sha256') != req_sha
            or matches[0].get('output_dir') != request['output_dir']
            or matches[0].get('vector_steps') != cohort['vector_steps']
            or session['request_identities'].get(req_sha) != req_canonical
            or receipt.get('schema') != 'roborender.bc-cohort-execution.v1' or receipt.get('verdict') != 'PASS'
            or receipt.get('cohort_manifest_raw_sha256') != cohort_binding['raw_sha256']
            or receipt.get('request_sha256') != req_sha or receipt.get('request_canonical_sha256') != req_canonical
            or receipt.get('session_execution_canonical_sha256') != digest(canonical(session))
            or receipt.get('collector_execution') != execution
            or receipt.get('actual_num_envs') != len(request['motions'])
            or receipt.get('actual_vector_steps') != cohort['vector_steps']):
        raise ValueError('cohort execution does not match completed request/session')
    return bindings, docs

def assemble(*, source_pool, sampling_plan, motion_inventory, native_motion_eligibility,
             native_source_root, native_source_manifest, render_bindings, capacity_receipts, output):
    if Path(output).exists():
        raise FileExistsError('protocol output already exists')
    pool, plan = bound(source_pool), bound(sampling_plan)
    if plan.get('schema') != 'roborender.coverage-sampling-plan.v3':
        raise ValueError('this successor requires train/validation plus external success-test v3')
    source, models, adapter = source_expectations(native_source_root, native_source_manifest)
    if not capacity_receipts or any(bound(b).get('schema') != 'roborender.native-bc-capacity.v2' for b in capacity_receipts):
        raise ValueError('guarded production handoff requires actual bounded-buffer v2 capacity')
    root = Path(source_pool['path']).resolve().parent
    cohorts = [c for part in plan['partitions'].values() for c in part['cohorts']]
    ids = [c['cohort_id'] for c in cohorts]
    existing = {p.name for p in (root / 'manifests/cohorts').iterdir() if p.is_dir()}
    if len(set(ids)) != len(ids) or existing != set(ids):
        raise ValueError('all and only planned cohort directories are required')
    cb, tb, eb, session_bindings, session_counts, requests_seen = [], [], [], {}, Counter(), {}
    execution = topology = None
    profiles, resources = {}, {}
    mixed_topology = False
    for cohort_id in ids:
        audit = root / 'manifests/cohorts' / cohort_id
        cb.append(binding(audit / 'native-cohort-manifest.json'))
        tb.append(binding(audit / 'transfer-receipt.json'))
        bindings, docs = execution_chain(root, cb[-1], source, native_source_manifest, adapter)
        eb.append(bindings['execution-receipt.json'])
        session, launch = docs['session-execution.json'], docs['launch.json']
        sid = session['session_id']
        if sid in session_bindings and session_bindings[sid] != bindings['session-execution.json']:
            raise ValueError('same session identity has different evidence')
        session_bindings[sid] = bindings['session-execution.json']
        session_counts[sid] += 1
        requests_seen.setdefault(sid, set()).add(bound(cb[-1])['request_sha256'])
        if execution is None:
            execution, topology = session['collector_execution'], launch.get('execution_topology')
        launch_topology = strict.validate_topology(launch.get('execution_topology'))
        if (execution != session['collector_execution']
                or {k: v for k, v in topology.items() if k != 'cgroup_limit_bytes_per_collector'}
                != {k: v for k, v in launch_topology.items() if k != 'cgroup_limit_bytes_per_collector'}):
            raise ValueError('mixed collector runtime/source/host/concurrency/reserve is not admitted')
        mixed_topology |= topology != launch_topology
        # The native receipt lives on Leo; the identical copied bytes live on
        # H200. Paths are host-local provenance, never the cross-host identity.
        capacity_sha = (launch.get('capacity_binding') or {}).get('raw_sha256')
        matched = [b for b in capacity_receipts if b['raw_sha256'] == capacity_sha]
        if len(matched) != 1 or bound(matched[0]).get('num_envs') != len(bound(cb[-1])['request']['motions']):
            raise ValueError('actual production launch used an unbound capacity receipt')
        if capacity_sha in profiles and profiles[capacity_sha] != launch_topology:
            raise ValueError('one capacity receipt cannot qualify two different resource profiles')
        profiles[capacity_sha] = launch_topology
        resource = dict(session_execution=bindings['session-execution.json'], launch=bindings['launch.json'],
                        host_result=bindings['host-result.json'], capacity_receipt_raw_sha256=capacity_sha)
        if sid in resources and resources[sid] != resource:
            raise ValueError('one session cannot change its qualified resource profile')
        resources[sid] = resource
    for sid, value in session_bindings.items():
        doc = bound(value)
        if session_counts[sid] != doc['scene_initializations'] or requests_seen[sid] != set(doc['request_identities']):
            raise ValueError('incomplete session coverage')
    rebuilt, _ = build_pool(plan,
        [{'document': bound(x), 'sha256': x['raw_sha256']} for x in cb],
        [{'document': bound(x), 'sha256': x['raw_sha256']} for x in tb],
        dataset_root=root, expected_total=plan['total_target_chunks'], verify_destination_stats=True)
    if rebuilt != pool or {k: v['sha256'] for k, v in pool['model_assets'].items()} != models:
        raise ValueError('source pool/model pins do not reproduce actual evidence')
    render = bound(render_bindings)
    if set(render) != RENDER_KEYS:
        raise ValueError('exact renderer producer protocol-bindings required')
    protocol = dict(schema=strict.SCHEMA_V3, purpose='production', expected_motion_count=295,
        training_chunk_batch_size=16, native_collector_sha256=strict.NATIVE_COLLECTOR_SHA256,
        motion_inventory=motion_inventory, sampling_plan=sampling_plan,
        model_sha256=models, native_motion_eligibility=native_motion_eligibility,
        capacity_receipts=capacity_receipts, source_pool=source_pool,
        cohort_manifests=cb, transfer_receipts=tb, cohort_execution_receipts=eb,
        session_execution_receipts=list(session_bindings.values()), collector_execution=execution,
        execution_topology=topology,
        capacity_equivalence=dict(minimum_full_flushes_per_lane=2, maximum_growth_fraction=.1,
                                  absolute_growth_slack_bytes=256 << 20),
        closed_loop_test_panel=plan['artifact_roles']['closed_loop_test']['panel'], **render)
    if mixed_topology:
        protocol.update(schema=strict.SCHEMA_V4,
            execution_topology={k: v for k, v in topology.items() if k != 'cgroup_limit_bytes_per_collector'},
            capacity_profiles=profiles, session_resource_bindings=list(resources.values()))
    # Validate a private temporary candidate before publishing the final artifact.
    import tempfile
    with tempfile.TemporaryDirectory(prefix='bc-protocol-check-') as temporary:
        candidate = publish(Path(temporary) / 'protocol.json', protocol)
        _, _, _, summary = strict.validate_protocol_inputs(candidate['path'], candidate['raw_sha256'])
    result = publish(output, protocol)
    return dict(schema='roborender.bc-protocol-assembly.v1', protocol=result,
        cohort_count=len(cb), session_count=len(session_bindings), metadata_only=True,
        production_admitted=False, post_render_admission_required=True, **summary)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source-pool', 'sampling-plan', 'motion-inventory', 'native-motion-eligibility',
                 'native-source-manifest', 'render-bindings'):
        parser.add_argument('--' + name, nargs=2, metavar=('PATH', 'RAW_SHA256'), required=True)
    parser.add_argument('--capacity-receipt', nargs=2, action='append', required=True)
    parser.add_argument('--native-source-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    values = {key: cli_binding(getattr(args, key)) for key in
        ('source_pool', 'sampling_plan', 'motion_inventory', 'native_motion_eligibility',
         'native_source_manifest', 'render_bindings')}
    print(json.dumps(assemble(**values, native_source_root=args.native_source_root,
        capacity_receipts=[cli_binding(x) for x in args.capacity_receipt], output=args.output), sort_keys=True))

if __name__ == '__main__':
    main()
