"""Fresh continuous BC data admission for the existing two-update B32 smoke.

No active dataset selector, held-out set, production checkpoint or GPU mutation.
The first64 sorted native emissions are an explicit diagnostic subset, not a
replacement for the full all-emission production pool.
"""
from __future__ import annotations

import json
from pathlib import Path

from pipeline.bc_rendering import render_bc_teacher_8gpu as render
from pipeline.bc_rendering.continuous_source import load_expected
from pipeline.bc_rendering.render_recipes import CURRENT_RECIPE,recipe_config
from pipeline.bc_rendering.validate_bc_teacher_render import validate_chunk

FIELDS=('fresh_continuous_pool','fresh_render_plan','fresh_rendered_root')
SCHEMA='roborender.fresh-continuous-bc32-smoke-input.v1'


def requested(args):
    supplied=[getattr(args,k,None) is not None for k in FIELDS]
    if any(supplied) and not all(supplied):
        raise ValueError('fresh smoke requires explicit pool, render plan and rendered root together')
    return all(supplied)


def physical(path,*,directory=False):
    p=Path(path)
    if not p.is_absolute() or p.resolve(strict=True)!=p or (not p.is_dir() if directory else not p.is_file()):
        raise ValueError('exact physical fresh-smoke path required')
    return p


def validate_render_plan(plan,jobs):
    if (plan.get('schema')!='roborender.bc_teacher_8gpu_plan.v3'
            or plan.get('status')!='complete' or plan.get('episode_count')!=64
            or plan.get('frame_count')!=64*81 or plan.get('chain_count')!=64
            or any(plan.get(k)!=v for k,v in recipe_config(CURRENT_RECIPE).items())
            or len(jobs)!=64 or len({j['source_npz'] for j in jobs})!=64):
        raise ValueError('exact completed independent64-render diagnostic required')
    expected={j['output_npz']:j for chain in render.build_render_chains(jobs) for j in chain['jobs']}
    actual=[]
    for filename in plan.get('lane_plan_paths',[]):
        path=physical(filename)
        lane=json.loads(path.read_bytes())
        if lane.get('config')!=plan['config'] or lane.get('schema')!='roborender.bc_teacher_lane_plan.v3':
            raise ValueError('render lane config/source differs')
        actual.extend(lane['jobs'])
    if (len(actual)!=64 or len({j['output_npz'] for j in actual})!=64
            or {j['output_npz']:j for j in actual}!=expected):
        raise ValueError('completed render jobs differ from the64 admitted native emissions')
    return expected


def validate_render_execution(document,render_path,expected):
    from pipeline.bc_training import render_execution as execution
    binding=document['config'].get('execution_identity')
    if not isinstance(binding,dict):
        raise ValueError('fresh smoke requires genuine identity-bound renderer execution')
    protocol=execution.bound(binding)
    execution.verify_render_asset_identity(protocol)
    produced=execution._verified_outputs(binding,protocol,render_path)
    if set(produced)!=set(expected):
        raise ValueError('fresh worker execution does not cover the exact64 source/output pairs')
    return binding


def plan(args,*,source_identity):
    from pipeline.bc_training import qualify_raven_bc32_lifecycle as smoke
    from pipeline.bc_training import train_bc_h200 as bc
    if not requested(args):raise ValueError('explicit fresh smoke inputs required')
    pool=physical(args.fresh_continuous_pool,directory=True)
    rendered=physical(args.fresh_rendered_root,directory=True)
    render_path=physical(args.fresh_render_plan)
    rows=load_expected(pool/'manifest.json',pool/'source-seal.json')
    if len(rows)<64 or rendered==pool/'isaac':
        raise ValueError('fresh smoke requires at least64 unique complete native chunks and separate generated output')
    jobs=render.discover_jobs(limit=64,source_root=pool/'isaac',pool_manifest=pool/'manifest.json',
        pool_seal=pool/'source-seal.json',output_root=rendered,verify_source_hashes=True)
    document=json.loads(render_path.read_bytes())
    if document.get('source_root')!=str(pool/'isaac') or document.get('output_root')!=str(rendered):
        raise ValueError('render execution belongs to another source/output pool')
    contexts=validate_render_plan(document,jobs)
    execution_binding=validate_render_execution(document,render_path,contexts)
    config=document['config']
    for key,value in dict(num_inference_steps=50,cfg_scale=5.0,height=240,width=416,
            student_height=108,student_width=192,seed=42,adapter_sha256=render.ADAPTER_SHA256,
            prompt_sha256=render.sha256_file(render.RAVEN_RENDER_PROMPT)).items():
        if config.get(key)!=value:raise ValueError('fresh BC renderer recipe changed: '+key)
    bootstrap=args.asset_root/'checkpoints/students/bootstrap/g1_3view_student_bootstrap_v1.pt'
    spec=bootstrap.with_suffix('.policy_spec.json')
    if (bc.sha256_file(bootstrap)!=smoke.BOOTSTRAP_SHA or bc.sha256_file(spec)!=smoke.SPEC_SHA
            or json.loads(spec.read_bytes()).get('checkpoint_sha256')!=smoke.BOOTSTRAP_SHA):
        raise ValueError('exact canonical saved bootstrap/spec required')
    selected=[]
    for job in jobs:
        output=physical(job['output_npz']);source=Path(job['source_npz'])
        errors=validate_chunk(source,output,chain_context=contexts[str(output)],deep_rgb=True)
        if errors:raise ValueError('invalid fresh rendered/teacher pair: '+str(errors[:3]))
        selected.append(dict(path=str(output),chunk_id=source.stem,original_role='fresh_continuous_train',
            size_bytes=output.stat().st_size,source_payload_sha256=job['source_payload_sha256'],
            rendered_payload_sha256=bc.sha256_file(output)))
    bindings={name:bc.sha256_file(pool/name) for name in ('manifest.json','source-seal.json','chunks.jsonl',
        'evidence/native-manifest.json','evidence/execution.json','evidence/host-result.json','evidence/launch.json')}
    return dict(schema=smoke.SCHEMA,execute=args.execute,optimizer_updates=2,
        chunks_per_update=32,frames_per_chunk=81,policy_forward_population=2592,
        student_lifecycle=bc.REPLAY_STUDENT_LIFECYCLE,batch_schema=bc.RAVEN_REPLAY_B32_BATCHING_CONTRACT,
        validation_enabled=False,completed_epoch=False,production_checkpoint=False,production_promoted=False,
        success_rate_tested=False,compute_release_authorized=False,full_training_chunk_count=len(rows),
        full_epoch_updates=None,dataset_root=str(pool),source_identity=source_identity,seed=args.seed,
        bootstrap=str(bootstrap),bootstrap_sha256=smoke.BOOTSTRAP_SHA,policy_spec=str(spec),policy_spec_sha256=smoke.SPEC_SHA,
        selected_chunks=selected,selected_origin_counts={'fresh_continuous_train':64},
        selection='diagnostic_first64_native_emissions_no_outcome_filter',production_sorted_cursor_exercised=False,
        heldout_validation_examples=0,initial_checkpoint_is_bc116=False,random_constructor_weights_retained=False,
        initial_policy='strict_full_state_load_from_saved_canonical_initial_student_checkpoint',
        fresh_input_protocol=dict(schema=SCHEMA,native_pool=bindings,render_plan_path=str(render_path),
            render_plan_sha256=bc.sha256_file(render_path),renderer_execution_identity=execution_binding,
            retired_data_used=False,training_subset_only=True),
        storage_registry=str(args.registry),storage_registry_sha256=bc.sha256_file(args.registry))
