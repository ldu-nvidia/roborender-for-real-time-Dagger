"""Additive 2560/320 admission over the separately admitted legacy2000 pool.

This opt-in contract preserves historical unknown provenance rather than
pretending this is an all-fresh dataset. Only metadata and file sizes are read;
native transfer/execution and genuine independent renderer receipts are
reconstructed before the new indexes may authorize learning.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import re

from pipeline.bc_training import legacy_reuse as legacy
from pipeline.bc_training.handoff_common import binding, bound, canonical, digest, publish, read
from pipeline.bc_training.assemble_production_protocol import execution_chain
from pipeline.bc_training.production_protocol import validate_bounded_capacity, valid_sha
from pipeline.legacy.bc_collection.transfer_native_cohort import selected_outputs
from pipeline.legacy.bc_collection.artifact_roles import panel_reference
from pipeline.deployment.rtx_actor.isaac6_bc_production import (
    SCHEMA_V2 as REQUEST_SCHEMA, NATIVE_COLLECTOR_SHA256, source_run_tag,
    selected_chunk_ordinals, validate_collection_request,
)

PLAN_SCHEMA = 'roborender.bc-training-extension-admission-plan.v1'
ADMISSION_SCHEMA = 'roborender.bc-training-extension-admission.v1'
POOL_SCHEMA = 'roborender.bc-training-extension-population.v1'
SAMPLING_SCHEMA = 'roborender.bc-training-extension560-plan.v1'
BASE_ADMISSION_SHA256 = '6611f8fb32b03ebb34839fde418b9865db5355aa67b69ccff3a1b7ccbea9734e'
SAMPLING_PLAN_SHA256 = '3b514ade112e4878d7db5dee2e5f3996f91bd8d9a8ae6a39997c23fd0a08ec6e'
NATIVE_SOURCE_SHA256 = '4c441616e2a9257d7a064d72f7481c443adda855e2e6e5319e6b871cc35151f1'
COUNTS = {'train':2560, 'validation':320}
LIMITATIONS = legacy.LIMITATIONS + [
    'The new560 cover all295 motions; legacy2000 motion/reset frequencies remain unknown.',
    'Disjoint recorded process seeds do not prove physical reset nonduplication against unknown legacy history.',
]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def absolute(path):
    path = Path(path)
    require(path.is_absolute() and '..' not in path.parts, 'canonical absolute dataset path required')
    return path.resolve()


def base_report(reference):
    require(reference.get('raw_sha256') == BASE_ADMISSION_SHA256,
            'only the separately admitted immutable2000/320 base is eligible')
    report = bound(reference)
    require(report.get('schema') == legacy.ADMISSION_SCHEMA,
            'base must be the distinct legacy-reuse admission')
    plan = bound(report['plan'])
    return legacy.validate_training_admission(reference['path'], report['split_files']['train']['path'],
        report['split_files']['validation']['path'], allow_legacy_reuse=True, run_name=report['run_name'],
        bootstrap_path=report['bootstrap_path'], policy_spec_path=report['bootstrap_spec']['path'],
        dataset_root=plan['dataset_root'])


def sampling_plan(reference, base):
    require(reference.get('raw_sha256') == SAMPLING_PLAN_SHA256,
            'the reviewed immutable560 sampling plan is required')
    plan = bound(reference)
    base_plan = bound(base['plan'])
    validation = bound(base_plan['fresh_validation_plan'])
    require(plan.get('schema') == SAMPLING_SCHEMA and plan.get('seed') == 2026101521
        and plan.get('validation_plan') == base_plan['fresh_validation_plan']
        and plan.get('legacy_manifest') == base_plan['legacy_manifest']
        and plan.get('closed_loop_test_panel') == base_plan['closed_loop_test_panel']
        and plan.get('total_target_chunks') == 560 and plan.get('frames_per_chunk') == 81
        and plan.get('exposure_span_chunks') == 9 and plan.get('num_envs_ceiling') == 6
        and plan.get('selection_uses_labels_rewards_or_loss') is False
        and set(plan.get('partitions',{})) == {'train'}, 'extension sampling references or quotas differ')
    cohorts = plan['partitions']['train']['cohorts']
    slots = [slot for c in cohorts for slot in c['slots']]
    require(len(cohorts)==50 and len({c['cohort_id'] for c in cohorts})==50
        and len(slots)==295 and len({s['motion_id'] for s in slots})==295
        and Counter(s['target_chunks'] for s in slots)=={2:265,1:30}
        and sum(c['target_chunks'] for c in cohorts)==560, 'exact560/295 motion population required')
    motions = {s['motion_id']:s['motion_sha256'] for s in slots}
    require(motions == base_plan['target_motion_bindings'], 'training and held-out motion bytes differ')
    seeds = [c['process_seed'] for c in cohorts]
    forbidden = set(base_plan['historical']['known_process_seeds'])
    forbidden.update(c['process_seed'] for c in validation['partitions']['validation']['cohorts'])
    forbidden.update(base['closed_loop_test']['simulator_seeds'])
    require(len(set(seeds))==50 and not set(seeds)&forbidden
        and all(type(s) is int and 0<=s<2**31 for s in seeds),
        'extension seeds overlap known training/validation/test seeds')
    for c in cohorts:
        require(c['split']=='train' and c['num_envs']==len(c['slots']) and c['num_envs'] in (1,6)
            and c['target_chunks']==sum(s['target_chunks'] for s in c['slots']), 'invalid fixed-B training cohort')
        for s in c['slots']:
            require(s['exposure_span_chunks']==9 and len(s['selected_chunk_ordinals'])==s['target_chunks']
                and s['selected_chunk_ordinals']==sorted(set(s['selected_chunk_ordinals']))
                and all(type(i) is int and 0<=i<9 for i in s['selected_chunk_ordinals']),
                'invalid preselected complete-chunk ordinals')
    return plan


def _plan_value(*, run_name, base_admission, extension_plan, dataset_root, base):
    require(isinstance(run_name,str) and re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{1,95}',run_name)
        and not run_name.startswith('smoke-'), 'explicit new production experiment name required')
    extension = sampling_plan(extension_plan,base)
    root = absolute(dataset_root)
    require(Path(bound(base['plan'])['dataset_root']).is_relative_to(root),
            'combined dataset root must contain preserved base artifacts')
    return dict(schema=PLAN_SCHEMA,purpose='legacy_training_extension',run_name=run_name,
        base_admission=base_admission,extension_plan=extension_plan,dataset_root=str(root),
        counts=COUNTS,added_training_chunks=560,training_chunk_batch_size=16,training_batches_per_epoch=160,
        bootstrap_path=base['bootstrap_path'],bootstrap_sha256=base['bootstrap_sha256'],
        bootstrap_spec=base['bootstrap_spec'],initialization='clean_bootstrap_only',
        resume_policy='same_admission_and_sampling_identity_only',
        closed_loop_test=base['closed_loop_test'],motion_inventory=extension['motion_inventory'],
        native_source_sha256=NATIVE_SOURCE_SHA256,renderer_recipe='raven_3view_50step_independent_v2',
        existing_admission_unchanged=True,limitations=LIMITATIONS,
        planning_only=True,production_admitted=False,training_extension_admitted=False,
        historical_motion_frequencies_known=False,new_training_motion_count=295)


def build_plan(*, run_name, base_admission, extension_plan, dataset_root):
    return _plan_value(run_name=run_name,base_admission=base_admission,extension_plan=extension_plan,
        dataset_root=dataset_root,base=base_report(base_admission))


def validated_context(plan):
    require(plan.get('schema')==PLAN_SCHEMA,'distinct additive training-extension plan required')
    base=base_report(plan['base_admission'])
    expected = _plan_value(**{k:plan[k] for k in ('run_name','base_admission','extension_plan','dataset_root')},base=base)
    require(plan==expected,'additive training-extension plan differs from pinned inputs')
    return base,sampling_plan(plan['extension_plan'],base)


def validate_plan(plan):
    validated_context(plan)
    return plan


def native_pool(plan,cohorts,transfers,*,dataset_root):
    """Reconstruct the exact frozen native560 pool using complete row ledgers."""
    planned = {c['cohort_id']:c for c in plan['partitions']['train']['cohorts']}
    docs = {c['document']['request']['cohort_id']:c for c in cohorts}
    moved = {c['document']['cohort_id']:c for c in transfers}
    require(len(planned)==50 and len(docs)==len(cohorts) and len(moved)==len(transfers)
        and set(docs)==set(planned)==set(moved), 'exact50 native cohorts and verified transfers required')
    root=absolute(dataset_root); plan_sha=digest(canonical(plan))
    output=[]; excluded=[]; audits=[]; hashes=set(); all_hashes=set(); names=set(); keys=set(); ends={}; models=None
    motions={}; seeds=set()
    for batch,(cid,p) in enumerate(planned.items()):
        env=docs[cid]; doc=env['document']; t=moved[cid]['document']
        request=validate_collection_request(doc['request']); slots=p['slots']
        require(doc.get('schema')=='roborender.bc-native-cohort.v1' and doc.get('status')=='complete'
            and doc.get('teacher_only') is True and doc.get('optimizer_steps')==0
            and doc.get('native_payload_writer_unchanged') is True
            and doc.get('native_collector_sha256')==NATIVE_COLLECTOR_SHA256
            and request['schema']==REQUEST_SCHEMA and request['split']=='train'
            and request['run_id']=='bc-leo-training560-20261015'
            and request['sampling_plan_sha256']==plan_sha and request['cohort_id']==cid
            and request['process_seed']==p['process_seed'] and request['process_seed'] not in seeds
            and request['process_seed'] not in plan['all_forbidden_process_seeds']
            and request['chunks_per_env']==[s['target_chunks'] for s in slots]
            and selected_chunk_ordinals(request)==[s['selected_chunk_ordinals'] for s in slots]
            and request['exposure_span_chunks_per_env']==[s['exposure_span_chunks'] for s in slots]
            and [(m['motion_id'],m['sha256']) for m in request['motions']]==[(s['motion_id'],s['motion_sha256']) for s in slots]
            and len(slots)==p['num_envs'] and type(doc['vector_steps']) is int
            and doc['vector_steps']>0 and doc['vector_steps']%81==0
            and doc['vector_steps']<=request['max_windows']*81,
            'native training cohort changed declared assignment or completion')
        seeds.add(request['process_seed'])
        assets=doc.get('model_assets',{})
        require(set(assets)=={'teacher','homie_walk','homie_stand'}
            and all(set(a)=={'path','sha256'} and Path(a['path']).is_absolute() and valid_sha(a['sha256'])
                    for a in assets.values()),'complete teacher/Homie model identities required')
        if models is None: models=assets
        require(models==assets,'mixed teacher/Homie assets')
        chosen_outputs=selected_outputs(doc,t['remote_runtime'])
        require(t.get('schema')=='roborender.bc-cohort-transfer.v1' and t.get('verified') is True
            and t.get('cohort_manifest_sha256')==env['sha256'] and len(t['files'])==len(chosen_outputs),
            'verified exact native payload transfer required')
        tf={r['source_path']:r for r in t['files']}
        require(len(tf)==len(t['files']),'duplicate transferred source')
        emitted=Counter(); chosen=Counter()
        for row in doc['chunks']:
            lane=row['local_env_id']; require(type(lane) is int and 0<=lane<len(slots),'invalid native lane')
            slot=slots[lane]; key=(row['source_run_tag'],row['rank'],lane,row['seq'])
            episode=(row['source_run_tag'],row['rank'],lane,row['episode_id'])
            require(key not in keys and row['source_run_tag']==source_run_tag(request)
                and row['rank']==request['rank'] and row['global_env_id']==request['global_env_offset']+lane
                and row['simulator_seed']==request['process_seed'] and row['split']=='train'
                and row['sampling_plan_sha256']==plan_sha and row['request_sha256']==doc['request_sha256']
                and row['motion_id']==slot['motion_id'] and row['motion_sha256']==slot['motion_sha256']
                and row['teacher_only'] is True and row['frames']==81
                and type(row['seq']) is int and row['seq']>=0
                and isinstance(row['episode_id'],str) and bool(row['episode_id'])
                and type(row['first_episode_step_index']) is int and row['first_episode_step_index']>=0
                and row['last_episode_step_index']-row['first_episode_step_index']==80
                and row['first_episode_step_index']>ends.get(episode,-1)
                and row.get('emitted_chunk_ordinal')==emitted[lane]
                and row.get('selection_policy')=='explicit_preselected_ordinals_v2'
                and valid_sha(row.get('source_sha256')) and type(row.get('source_bytes')) is int and row['source_bytes']>0,
                'native chunk identity or chronological episode interval differs')
            keys.add(key); ends[episode]=row['last_episode_step_index']
            require(row['source_sha256'] not in all_hashes,'duplicate native source payload')
            all_hashes.add(row['source_sha256'])
            select=emitted[lane] in slot['selected_chunk_ordinals']; emitted[lane]+=1
            require(row['selected_for_split'] is select and row['overcollection'] is (not select),
                    'selection changed predeclared ordinal')
            if not select: excluded.append(row); continue
            chosen[lane]+=1; r=tf.get(row['source_path'])
            require(r is not None and r['source_sha256']==row['source_sha256']==r['destination_sha256']
                and r['source_bytes']==row['source_bytes']==r['destination_bytes'], 'native transfer byte identity differs')
            path=Path(r['destination_path'])
            require(not path.is_symlink() and path.is_file() and path.parent==root/'isaac'
                and path.name==Path(row['source_path']).name and path.stat().st_size==row['source_bytes']
                and row['source_sha256'] not in hashes and path.name not in names,
                'native payload destination changed or duplicated')
            hashes.add(row['source_sha256']); names.add(path.name)
            require(motions.setdefault(row['motion_id'],row['motion_sha256'])==row['motion_sha256'],'motion bytes changed')
            output.append({**row,'chunk_id':f'trainext_chunk_{len(output):06d}','source_batch':batch,
                'sequence_id':row['seq'],'source_size_bytes':row['source_bytes'],'collection_filename':path.name,
                'isaac_npz':'isaac/'+path.name,'rendered_npz':'rendered_independent_v2/'+path.name,
                'cohort_manifest_sha256':env['sha256'],'transfer_receipt_sha256':moved[cid]['sha256']})
        require(doc['selected_chunks']==sum(chosen.values())
            and doc['overcollection_chunks']==sum(emitted.values())-sum(chosen.values())
            and doc['emitted_per_lane']==[emitted[i] for i in range(len(slots))]
            and [chosen[i] for i in range(len(slots))]==request['chunks_per_env'],
            'incomplete or changed native emitted/selected quotas')
        audits.append(dict(cohort_id=cid,manifest_sha256=env['sha256'],transfer_receipt_sha256=moved[cid]['sha256']))
    require(len(output)==560 and len(motions)==295,'full560/295 native motion pool incomplete')
    return dict(schema='roborender.bc_pool.native-cohorts.v1',status='sealed',frames_per_chunk=81,
        chunk_count=560,sampling_plan_sha256=plan_sha,partition_counts={'train':560},motion_count=295,
        chunks=output,model_assets=models,native_collector_sha256=NATIVE_COLLECTOR_SHA256,motion_bindings=motions,
        cohort_audit=audits,overcollection_excluded=excluded,selection_policy=plan['selection_policy'],
        collection_cost=plan['collection_cost'],sampling_plan_schema=plan['schema'],
        collection_purpose='fresh_training_extension_only',payloads_read=False,production_admitted=False)


def native_execution(evidence_binding,sampling,base):
    evidence=bound(evidence_binding)
    require(evidence.get('schema')=='roborender.training560-native-execution.v1'
        and evidence.get('selected_chunks')==560 and evidence.get('sampling_plan')==base['extension_plan'],
        'genuine560 native execution evidence required')
    source=bound(evidence['source_pool']); root=Path(evidence['source_pool']['path']).parent
    native_source=bound(evidence['native_source_manifest'])
    require(native_source.get('source_sha256')==NATIVE_SOURCE_SHA256
        and digest(canonical(native_source['files']))==NATIVE_SOURCE_SHA256,'native source release differs')
    by_path={r['path']:r['sha256'] for r in native_source['files']}
    adapter=by_path['pipeline/deployment/rtx_actor/isaac6_bc_production.py']
    cohorts=[]; transfers=[]; sessions={}; counts=Counter(); execution=None; topology=None
    capacities={bound(b)['num_envs']:(b,bound(b)) for b in evidence['capacity_receipts']}
    require(len(capacities)==len(evidence['capacity_receipts'])==2 and set(capacities)=={1,6},
            'exact B1 and B6 capacity receipts required without duplicates')
    ids=[c['cohort_id'] for c in sampling['partitions']['train']['cohorts']]
    require({p.name for p in (root/'manifests/cohorts').iterdir() if p.is_dir()}==set(ids),
            'all and only50 planned native cohort directories required')
    for cid in ids:
        audit=root/'manifests/cohorts'/cid
        cb=binding(audit/'native-cohort-manifest.json'); tb=binding(audit/'transfer-receipt.json')
        doc=bound(cb); cohorts.append(dict(document=doc,sha256=cb['raw_sha256']))
        transfers.append(dict(document=bound(tb),sha256=tb['raw_sha256']))
        bindings,docs=execution_chain(root,cb,native_source,evidence['native_source_manifest'],adapter)
        s,l=docs['session-execution.json'],docs['launch.json']; sid=s['session_id']
        require(sid not in sessions or sessions[sid]==bindings['session-execution.json'],'mixed session identity')
        sessions[sid]=bindings['session-execution.json']; counts[sid]+=1
        if execution is None: execution=s['collector_execution']; topology=l['execution_topology']
        require(execution==s['collector_execution'] and topology==l['execution_topology'],'mixed native execution/topology')
        cap=capacities.get(len(doc['request']['motions']))
        require(cap is not None and l.get('capacity_binding',{}).get('raw_sha256')==cap[0]['raw_sha256'],
                'native launch used an unbound capacity receipt')
    require(len(sessions)==14 and all(counts[sid]==bound(b)['scene_initializations'] for sid,b in sessions.items()),
            'native session coverage incomplete')
    require(topology.get('host_id')=='leo-rtx-6k' and topology.get('concurrent_collectors')==1
        and topology.get('minimum_host_reserve_bytes',0)>=8<<30
        and topology.get('cgroup_limit_bytes_per_collector',44<<30)<=43<<30,
        'unqualified serial native topology')
    rebuilt=native_pool(sampling,cohorts,transfers,dataset_root=root)
    require(source==rebuilt,'native560 pool differs from exact collection/transfer evidence')
    models={k:v['sha256'] for k,v in source['model_assets'].items()}
    base_report_value=bound(base['base_admission']); base_plan=bound(base_report_value['plan'])
    require(models==base_plan['historical']['model_sha256'],'new teacher/Homie models differ from admitted base')
    protocol=dict(model_sha256=models,collector_execution=execution,execution_topology=topology,
        capacity_equivalence=dict(minimum_full_flushes_per_lane=2,maximum_growth_fraction=.10,
                                  absolute_growth_slack_bytes=256<<20))
    require(set(capacities)=={len(c['document']['request']['motions']) for c in cohorts},'native capacity coverage incomplete')
    for _,receipt in capacities.values(): validate_bounded_capacity(receipt,protocol)
    require(evidence['collector_execution']==execution and evidence['execution_topology']==topology
        and evidence['model_sha256']==models and len(evidence['session_execution_receipts'])==len(sessions)
        and {canonical(b) for b in evidence['session_execution_receipts']}=={canonical(b) for b in sessions.values()},
        'native execution summary differs from actual receipt chain')
    seal=bound(evidence['source_seal'])
    require(seal==dict(schema='roborender.bc-source-seal.v2',verdict='PASS',isaac_chunks=560,rendered_chunks=0,
        sampling_plan_sha256=digest(canonical(sampling)),payloads_read=False,
        qualification_scope='native provenance/exact560/verified transfer; render/admission pending',
        manifest_sha256=evidence['source_pool']['raw_sha256']), 'native source seal differs')
    return source


def scoped_episode(row):
    return (row['source_run_tag'],row['rank'],row['local_env_id'],row['episode_id'])


def assert_independence(fresh,base,rendered):
    """Use actual recorded scopes; native episode strings are only lane-local."""
    old_plan=bound(base['plan']); old_rows=bound(old_plan['legacy_manifest'])['chunks']
    val=bound(base['native_evidence']['source_pool'])
    val_rows=val['chunks']+val.get('overcollection_excluded',[])
    fresh_rows=fresh['chunks']+fresh.get('overcollection_excluded',[])
    require(not {r['source_sha256'] for r in fresh_rows}&{r['source_sha256'] for r in old_rows+val_rows},
            'new training raw source leaks to prior training or validation')
    require(not {scoped_episode(r) for r in fresh_rows}&{scoped_episode(r) for r in val_rows},
            'new training scoped episode leaks into held-out validation')
    require(not {(r['source_run_tag'],r['rank']) for r in fresh_rows}
        &{(r['source_run_tag'],r['rank']) for r in old_rows+val_rows}, 'new training reuses a prior process scope')
    known={r['simulator_seed'] for r in old_rows+val_rows if r.get('simulator_seed') is not None}
    panel=base['closed_loop_test']; known.update(panel['simulator_seeds'])
    require(not {r['simulator_seed'] for r in fresh_rows}&known,'new training known process seed leaks to validation/test')
    val_render=bound(base['rendering']['validation']['rendered_manifest'])['chunks']
    old_render=bound(base['rendering']['train']['rendered_manifest'])['chunks']
    require(not {r['rendered_sha256'] for r in rendered['chunks']}
        &{r['rendered_sha256'] for r in val_render+old_render}, 'new rendered payload duplicates an admitted payload')
    require(not {r['render_chain_id'] for r in rendered['chunks'] if r.get('render_chain_id')}
        &{r['render_chain_id'] for r in val_render if r.get('render_chain_id')},
        'new render reference chain leaks into validation')
    support={r['motion_id']:r['motion_sha256'] for r in fresh['chunks']}
    panel_reference(panel['panel'],inventory_sha256=panel['motion_inventory_sha256'],
        motion_bindings=support,forbidden_seeds=known-set(panel['simulator_seeds'])|{r['simulator_seed'] for r in fresh_rows})


def assemble_population(plan,native_evidence,rendering):
    base,sampling=validated_context(plan)
    fresh=native_execution(native_evidence,sampling,plan)
    require(set(rendering)=={'protocol','render_plan','rendered_manifest','execution_receipt'},
            'exact new560 render evidence required')
    evidence=bound(native_evidence)
    require(bound(rendering['protocol']).get('source_pool')==evidence['source_pool'],
            'renderer is bound to a different native pool')
    require(bound(rendering['execution_receipt']).get('model_sha256')
        ==bound(base['rendering']['train']['execution_receipt']).get('model_sha256')
        ==bound(base['rendering']['validation']['execution_receipt']).get('model_sha256'),
        'new renderer model/source/prompt identities differ from the admitted base')
    rendered=legacy._rendered_population(rendering,{r['source_sha256']:r for r in fresh['chunks']})
    assert_independence(fresh,base,rendered)
    root=absolute(plan['dataset_root']); base_root=Path(bound(base['plan'])['dataset_root'])
    base_population=bound(dict(path=base['source_manifest'],raw_sha256=base['source_manifest_sha256']))
    output=[]; paths=set(); hashes=set()
    for row in base_population['chunks']:
        path=(base_root/row['rendered_npz']).resolve(strict=True)
        require(path.is_relative_to(root) and path not in paths and row['rendered_sha256'] not in hashes,
                'base rendered path/hash escaped or duplicated')
        paths.add(path); hashes.add(row['rendered_sha256'])
        output.append({**row,'rendered_npz':str(path.relative_to(root))})
    for row in rendered['chunks']:
        path=(Path(rendered['rendered_pool'])/Path(row['rendered_npz']).name).resolve(strict=True)
        require(path.is_relative_to(root) and path.is_file() and path.stat().st_size==row['rendered_size_bytes']
            and path not in paths and row['rendered_sha256'] not in hashes,'new rendered path/size/hash differs or duplicates')
        paths.add(path); hashes.add(row['rendered_sha256'])
        output.append(dict(chunk_id='train-extension-'+row['chunk_id'],original_chunk_id=row['chunk_id'],
            split='train',rendered_npz=str(path.relative_to(root)),source_sha256=row['source_sha256'],
            rendered_sha256=row['rendered_sha256'],frames=81,
            stream_id=digest(canonical([row['source_run_tag'],row['rank'],row['local_env_id']])),
            provenance_origin='fresh_native_training_extension',source_rendered_manifest=rendering['rendered_manifest']['path'],
            source_rendered_manifest_sha256=rendering['rendered_manifest']['raw_sha256']))
    require(Counter(r['split'] for r in output)==COUNTS and len({r['chunk_id'] for r in output})==2880
        and len({r['source_sha256'] for r in output})==2880,'combined2560/320 population duplicates or differs')
    require(not {r['stream_id'] for r in output if r['split']=='train'}
        &{r['stream_id'] for r in output if r['split']=='validation'},'combined train/validation stream overlap')
    return dict(schema=POOL_SCHEMA,plan_canonical_sha256=digest(canonical(plan)),chunk_count=2880,
        frames_per_chunk=81,counts=COUNTS,chunks=output,closed_loop_test=plan['closed_loop_test'],
        limitations=LIMITATIONS,new_training_motion_count=295,historical_motion_frequencies_known=False,
        payloads_opened=False)


def split_documents(population,manifest_binding,*,split_dir):
    require(population.get('schema')==POOL_SCHEMA and population.get('counts')==COUNTS,
            'distinct2560/320 combined population required')
    return {role:dict(schema='roborender.bc_split.v2',split=role,chunk_count=count,
        source_manifest=os.path.relpath(manifest_binding['path'],absolute(split_dir)),
        source_manifest_sha256=manifest_binding['raw_sha256'],
        items=[r for r in population['chunks'] if r['split']==role]) for role,count in COUNTS.items()}


def audit_admission(*,plan_binding,native_evidence,rendering,population_binding,split_files):
    plan=bound(plan_binding); actual=assemble_population(plan,native_evidence,rendering)
    require(bound(population_binding)==actual and set(split_files)==set(COUNTS),'combined population differs from actual evidence')
    for role,b in split_files.items():
        require(bound(b)==split_documents(actual,population_binding,split_dir=Path(b['path']).parent)[role],
                'extension split differs from exact admitted population')
    return dict(schema=ADMISSION_SCHEMA,purpose='legacy_training_extension',run_name=plan['run_name'],
        training_extension_admitted=True,production_admitted=False,base_admission=plan['base_admission'],
        plan=plan_binding,native_evidence=native_evidence,rendering=rendering,
        source_manifest=population_binding['path'],source_manifest_sha256=population_binding['raw_sha256'],
        split_files=split_files,counts=COUNTS,training_batches_per_epoch=160,limitations=LIMITATIONS,
        initialization='clean_bootstrap_only',resume_policy='same_admission_and_sampling_identity_only',
        bootstrap_path=plan['bootstrap_path'],bootstrap_sha256=plan['bootstrap_sha256'],bootstrap_spec=plan['bootstrap_spec'],
        closed_loop_test=plan['closed_loop_test'],payloads_opened=False,closed_loop_test_executed=False,
        new_training_motion_count=295,historical_motion_frequencies_known=False)


def validate_training_admission(report_path,train_path,validation_path,*,allow_training_extension,run_name,
        bootstrap_path,policy_spec_path,resume_checkpoint=None,resume_validation_state=None,dataset_root=None):
    require(allow_training_extension and not (resume_validation_state is not None and resume_checkpoint is None),
            'training extension requires explicit opt-in and clean bootstrap or exact-protocol resume')
    report=json.loads(read(report_path))
    require(report.get('schema')==ADMISSION_SCHEMA and isinstance(run_name,str) and bool(run_name)
        and (resume_checkpoint is not None or report.get('run_name')==run_name)
        and set(report.get('split_files',{}))==set(COUNTS)
        and Path(report['split_files']['train']['path']).resolve()==Path(train_path).resolve()
        and Path(report['split_files']['validation']['path']).resolve()==Path(validation_path).resolve()
        and Path(report['bootstrap_path']).resolve()==Path(bootstrap_path).resolve()
        and binding(policy_spec_path)==report.get('bootstrap_spec'),'extension training invocation differs from admission')
    plan=bound(report['plan'])
    require(dataset_root is not None and absolute(dataset_root)==Path(plan['dataset_root']),'extension dataset root differs')
    actual=audit_admission(plan_binding=report['plan'],native_evidence=report['native_evidence'],rendering=report['rendering'],
        population_binding=dict(path=report['source_manifest'],raw_sha256=report['source_manifest_sha256']),
        split_files=report['split_files'])
    require(report==actual,'extension admission differs from rederived execution evidence')
    return report


def validate_resume_identity(checkpoint,expected_admission):
    prior=checkpoint.get('data_admission',{})
    require(prior.get('schema')==ADMISSION_SCHEMA and prior.get('training_extension_admitted') is True
        and valid_sha(prior.get('report_sha256')) and prior['report_sha256']==expected_admission.get('report_sha256'),
        'training-extension resume must use the same admitted dataset/protocol')


def publish_admission(*,plan_binding,native_evidence,rendering,output_dir):
    root=absolute(output_dir)
    if root.exists(): raise FileExistsError('fresh additive admission output directory required')
    plan=bound(plan_binding); population=assemble_population(plan,native_evidence,rendering)
    root.mkdir(parents=True,exist_ok=False)
    manifest=publish(root/'population.json',population)
    documents=split_documents(population,manifest,split_dir=root)
    splits={role:publish(root/(role+'.json'),doc) for role,doc in documents.items()}
    report=audit_admission(plan_binding=plan_binding,native_evidence=native_evidence,rendering=rendering,
        population_binding=manifest,split_files=splits)
    admission=publish(root/'data-admission.json',report)
    return dict(admission=admission,split_files=splits,counts=COUNTS,training_batches_per_epoch=160,
        training_extension_admitted=True,production_admitted=False,historical_motion_frequencies_known=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--plan-inputs',type=Path)
    group.add_argument('--admission-inputs',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.plan_inputs is not None:
        result=dict(plan=publish(args.output,build_plan(**json.loads(read(args.plan_inputs)))),planning_only=True)
    else:
        result=publish_admission(**json.loads(read(args.admission_inputs)),output_dir=args.output)
    print(json.dumps(result,sort_keys=True))


if __name__=='__main__': main()
