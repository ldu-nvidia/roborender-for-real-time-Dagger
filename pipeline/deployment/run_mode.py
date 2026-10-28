"""One explicit smoke/production recipe for all five learning stages.

Default CPU-only staging uses the existing collection, BC, MSE campaign and
resident test contracts. --execute runs those stages against explicitly bound
retained infrastructure. It never provisions/releases GPUs or invents receipts.
Never switch the mode of an already-started run. This v1 recipe is historical:
validation execution requires explicit --allow-legacy-validation opt-in.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re

from pipeline.bc_training.handoff_common import binding, bound, canonical
from pipeline.dagger.beta_campaign import immutable_json
from pipeline.deployment.rtx_actor.production_success_contract import require_panel

SCHEMA = 'roborender.pipeline-mode.v1'
BETAS = [.75, .50, .25, 0.]


def recipe(mode):
    if mode not in ('smoke', 'production'):
        raise ValueError('mode must explicitly be smoke or production')
    smoke = mode == 'smoke'
    return dict(schema=SCHEMA, mode=mode,
        purpose='infrastructure_qualification' if smoke else 'policy_training',
        bc=dict(train_chunks=16 if smoke else 2560, validation_chunks=2 if smoke else 320,
            collection_environments=1 if smoke else 'qualified_collection_profile',
            offline_test_chunks=0, batch_chunks=16, frames_per_chunk=81,
            max_epochs=2 if smoke else 100, min_epochs=2 if smoke else 10,
            patience=2 if smoke else 8, validation_every_epochs=1,
            baseline_validation=True, fresh_collection_required=smoke,
            renderer_steps=50, renderer_cfg=5., max_chain=1),
        dagger=dict(rounds=4, teacher_probabilities=BETAS,
            environments_per_rtx=2 if smoke else 'capacity_qualified_profile',
            physical_rtx_gpus=[0, 1], h200_workers=4, rollout_steps=81,
            max_updates_per_round=2 if smoke else 'coverage_plan_budget',
            validation_chunks_per_round=4 if smoke else 320,
            validation_interval_updates=1 if smoke else 'coverage_plan_interval',
            minimum_validation_checks=2, patience=2 if smoke else 5,
            maximum_coverage_epochs=None if smoke else 10,
            baseline_validation=True, previous_round_replay=False,
            parent='previous_stage_absolute_best_validation_policy_and_optimizer',
            renderer_steps=5, renderer_cfg=1., teacache_threshold=.3),
        test=dict(cases=2 if smoke else 295, after_stages=['bc', 'dagger1', 'dagger2', 'dagger3', 'dagger4'],
            same_frozen_panel_all_stages=True, corrected_initial_reset=True,
            observation_mode='image3f_3view', optimizer_steps=0,
            selection_feedback=False, quality_claim_allowed=not smoke),
        execution=dict(fake_components_allowed=False, retain_compute=True,
            auto_release_compute=False, native_replay_allowed=False,
            keep_kit_warm=True, fresh_training_reset_after_test=True,
            full_environment_batch_forward=True, all_lane_barrier=True),
        logging=dict(project='dagger', mode='online', log_train_every_optimizer_update=True,
            log_validation_every_check=True, log_success_after_each_stage=True,
            separate_mode_namespaces=True))


def stage(mode, settings, output):
    """Freeze mode, paths and seeds. No silent production down-sizing."""
    p = recipe(mode)
    required = {'run_id', 'seed', 'inventory', 'test_panel', 'dataset_root', 'asset_root',
                'source_root', 'source_sha256', 'teacher_sha256', 'policy_spec_sha256',
                'rendered_observation_config'}
    if not required <= set(settings):
        raise ValueError('missing mode settings: ' + ','.join(sorted(required-set(settings))))
    run = settings['run_id']
    if (not isinstance(run, str) or not re.fullmatch(r'[a-z0-9][a-z0-9-]{1,39}', run)
            or (mode == 'smoke') != run.startswith('smoke-')):
        raise ValueError('use a fresh smoke-* identity only for smoke; maximum 40 characters')
    if type(settings['seed']) is not int or not 0 <= settings['seed'] < 2**31-1000000:
        raise ValueError('bounded explicit master seed required')
    for field in ('source_sha256', 'teacher_sha256', 'policy_spec_sha256'):
        if not re.fullmatch('[0-9a-f]{64}', settings[field]):
            raise ValueError('invalid immutable identity: ' + field)
    for field in ('dataset_root', 'asset_root', 'source_root'):
        if not Path(settings[field]).is_absolute() or '..' in Path(settings[field]).parts:
            raise ValueError('absolute normalized root required: ' + field)
    root = Path(output).resolve()
    original = require_panel(bound(settings['test_panel']), raw_sha256=settings['test_panel']['raw_sha256'])
    from pipeline.deployment.rendered_success_dispatch import configuration
    configuration(settings['rendered_observation_config'], original, '0'*64)
    inventory = bound(settings['inventory'])
    if mode == 'smoke':
        from pipeline.deployment.rtx_actor.smoke_success_contract import make_panel
        from pipeline.legacy.bc_collection.sampling_plan import build_offline_plan
        panel = make_panel(original)
        chosen = {r['motion_id']: r['sha256'] for r in panel['cohorts'][0]['assignment']['lanes']}
        rows = [r for r in inventory['motions'] if chosen.get(r['motion_id']) == r['sha256']]
        if len(rows) != 2:
            raise ValueError('smoke motions are absent from the bound inventory')
        subset = {**inventory, 'motion_count': 2, 'motions': rows,
                  'motion_records_sha256': hashlib.sha256(canonical(rows)).hexdigest()}
        inv_path = root/'smoke-inventory.json'
        immutable_json(inv_path, subset)
        inv_sha = hashlib.sha256(canonical(subset)).hexdigest()
        panel['motion_inventory_sha256'] = inv_sha
        panel['cohorts'][0]['assignment']['motion_inventory_sha256'] = inv_sha
        test_path = root/'success-panel.json'
        immutable_json(test_path, panel)
        collection = build_offline_plan(subset, seed=settings['seed'], num_envs=1,
            success_panel=binding(test_path), train_chunks=16, validation_chunks=2)
        immutable_json(root/'bc-sampling-plan.json', collection)
        collection_binding = binding(root/'bc-sampling-plan.json')
    else:
        # Existing reuse-first production data is not silently recollected or
        # repartitioned by a mode switch. Its independent admission is required.
        for name in ('bc_production_protocol', 'coverage_plan'):
            if name not in settings:
                raise ValueError('production requires the frozen ' + name)
            bound(settings[name])
        from pipeline.dagger.coverage_production import validate_plan,SCALED_SCHEMA
        coverage = validate_plan(bound(settings['coverage_plan']))
        if coverage['schema']==SCALED_SCHEMA:
            from pipeline.deployment.capacity_admission import validate as validate_capacity
            if 'capacity_admission' not in settings:raise ValueError('actual paired capacity admission required')
            validate_capacity(bound(settings['capacity_admission']),source_sha256=settings['source_sha256'],
                environments_per_actor=coverage['topology']['environments_per_actor'])
        if (coverage['source_sha256'] != settings['source_sha256']
                or coverage['panel'] != settings['test_panel']
                or coverage['stopping']['patience'] != 5):
            raise ValueError('production coverage/source/test/stopping binding differs')
        p['dagger'].update(environments_per_rtx=coverage['topology']['environments_per_actor'],
            max_updates_per_round=coverage['max_updates_per_round'],
            validation_interval_updates=coverage['updates_per_coverage_epoch'])
        test_path = Path(settings['test_panel']['path'])
        collection_binding = settings['bc_production_protocol']
    dagger_seeds = {settings['seed']+10000+100003*(rid-1)+gpu for rid in range(1,5) for gpu in (0,1)}
    test = json.loads(test_path.read_bytes())
    test_seeds = {c['assignment']['simulator_seed'] for c in test['cohorts']}
    if dagger_seeds & test_seeds:
        raise ValueError('train/test seed collision; choose another master seed')
    p.update(run_id=run, settings=deepcopy(settings), test_panel=binding(test_path),
        bc_population=collection_binding, source_sha256=settings['source_sha256'],
        source_root=settings['source_root'], output_root=str(root),
        dagger_seed=settings['seed']+10000,
        stages=['prepare_bc', 'render_bc', 'train_validate_bc', 'test_bc',
                'train_validate_test_dagger1', 'train_validate_test_dagger2',
                'train_validate_test_dagger3', 'train_validate_test_dagger4'],
        stage_names={name: f'{run}-{name}' for name in ('BC', 'dagger-teacher75',
            'dagger-teacher50', 'dagger-teacher25', 'dagger-teacher0')},
        runtime_certified=False, launched=False)
    immutable_json(root/'pipeline-mode.json', p)
    return p


def bc_job(plan):
    """Actual offline renderer/admission/trainer argv, not another learner."""
    from pipeline.bc_training.offline_pipeline import plan as offline_plan
    cfg, bc = plan['settings'], plan['bc']
    kwargs = {}
    if plan['mode'] == 'production':
        ref = cfg['bc_production_protocol']
        from pipeline.bc_training.training_extension import ADMISSION_SCHEMA
        if bound(ref).get('schema')==ADMISSION_SCHEMA:
            from pipeline.bc_training.reuse_pipeline import plan as reuse_plan
            result = reuse_plan(admission=ref,run_name=plan['stage_names']['BC'],
                source_root=cfg['source_root'],asset_root=cfg['asset_root'],
                runtime_root=Path(plan['output_root'])/'bc-execution',max_epochs=bc['max_epochs'],
                min_epochs=bc['min_epochs'],patience=bc['patience'],seed=cfg['seed'])
            result['stages'][-1]['argv'] += ['--training-protocol', 'legacy_validation']
            return result
        kwargs.update(production_protocol=ref['path'], production_protocol_sha256=ref['raw_sha256'])
    result = offline_plan(dataset_root=cfg['dataset_root'], asset_root=cfg['asset_root'],
        source_root=cfg['source_root'], runtime_root=Path(plan['output_root'])/'bc-execution',
        run_name=plan['stage_names']['BC'], gpus=(0,1,2,3),
        max_epochs=bc['max_epochs'], min_epochs=bc['min_epochs'], patience=bc['patience'],
        qualification=plan['mode']=='smoke', **kwargs)
    # Preserve the explicit experiment seed rather than the historical default.
    argv = result['stages'][-1]['argv']
    argv[argv.index('--seed')+1] = str(cfg['seed'])
    # v1 is an explicitly historical validation recipe. Its executor is gated;
    # do not silently reinterpret its checkpoint selection under fixed_final.
    argv += ['--training-protocol', 'legacy_validation']
    return result


def stage_dagger(plan, *, selected_checkpoint, checkpoint_sha256):
    """Call after actual BC selection; never use a pending or guessed parent."""
    from pipeline.dagger.services.contracts import file_sha256
    checkpoint = Path(selected_checkpoint).resolve(strict=True)
    if file_sha256(checkpoint) != checkpoint_sha256:
        raise ValueError('selected BC checkpoint identity differs')
    root = Path(plan['output_root'])/'dagger'
    cfg = plan['settings']
    if plan['mode'] == 'smoke':
        from pipeline.dagger.prepare_mse_smoke import prepare
        return prepare(root, run_id='dagger-leo-'+plan['run_id'], seed=plan['dagger_seed'],
            checkpoint=checkpoint, checkpoint_sha=checkpoint_sha256,
            panel_path=plan['test_panel']['path'], source_sha=cfg['source_sha256'],
            teacher_sha=cfg['teacher_sha256'], policy_spec_sha=cfg['policy_spec_sha256'],
            max_iterations=plan['dagger']['max_updates_per_round'], corrected_success=True,
            rendered_observation_config=cfg['rendered_observation_config'])
    from pipeline.dagger.coverage_production import stage as stage_production
    result=stage_production(bound(cfg['coverage_plan']), output_root=root,
        checkpoint=checkpoint, checkpoint_sha256=checkpoint_sha256,
        rendered_observation_config=cfg['rendered_observation_config'])
    return {**result,'config_path':result['first_round']['path'],'plan_path':result['mse_plan']['path']}


def test_command(plan, *, stage_name, checkpoint_sha256, panel_file, output_name,
                 rendered_config_file=None):
    """Native resident API command, identical reset semantics in both modes."""
    if stage_name not in ('bc', 'dagger1', 'dagger2', 'dagger3', 'dagger4'):
        raise ValueError('unknown learning stage')
    if not re.fullmatch('[0-9a-f]{64}', checkpoint_sha256):
        raise ValueError('selected checkpoint digest required')
    panel = bound(plan['test_panel'])
    from pipeline.deployment.rendered_success_dispatch import artifact, payload_fields
    _, _, config_sha = artifact(plan['settings'].get('rendered_observation_config'), panel, checkpoint_sha256)
    if rendered_config_file is None:
        raise ValueError('actual staged rendered configuration path required')
    if plan['mode'] == 'smoke':
        from pipeline.deployment.rtx_actor.smoke_success_contract import evaluation_options
        options = evaluation_options(panel)
    else:
        from pipeline.deployment.rtx_actor.production_success_contract import evaluation_options
        options = evaluation_options(panel, raw_sha256=plan['test_panel']['raw_sha256'])
    if Path(panel_file).is_absolute() or '..' in Path(panel_file).parts:
        raise ValueError('resident panel_file must be relative to its jobs root')
    if not re.fullmatch(r'[a-z0-9][a-z0-9-]{0,95}', output_name):
        raise ValueError('fresh bounded resident output identity required')
    return dict(gpu=0, command_id=output_name, operation='evaluate_explicit_panel', payload=dict(
        panel_file=panel_file, panel_sha256=plan['test_panel']['raw_sha256'],
        checkpoint_sha256=checkpoint_sha256, arm='bc' if stage_name=='bc' else 'dagger',
        output_name=output_name, **options, **payload_fields(rendered_config_file, config_sha)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('smoke', 'production'), required=True)
    parser.add_argument('--settings', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--selected-bc-checkpoint', type=Path)
    parser.add_argument('--selected-bc-sha256')
    parser.add_argument('--execute',action='store_true',help='Run the real bound site workflow; no allocation/release')
    parser.add_argument('--allow-legacy-validation', action='store_true',
        help='Explicit historical v1 validation compatibility; not the new warm no-validation pipeline')
    parser.add_argument('--execution-settings',type=Path,help='Pinned Leo/retained-H200 site and capacity references')
    args = parser.parse_args()
    if args.execute:
        from pipeline.deployment.training_launch_policy import require_training_launch
        require_training_launch(validation_enabled=True,
            allow_legacy_validation=args.allow_legacy_validation)
    if bool(args.selected_bc_checkpoint) != bool(args.selected_bc_sha256):
        parser.error('selected BC path and digest must be supplied together')
    p = stage(args.mode, json.loads(args.settings.read_bytes()), args.output)
    # A reconciled continuation has already admitted the original BC receipts.
    # Rebuilding a BC job under its fresh DAgger namespace would both fail the
    # original data admission and incorrectly suggest that BC should run again.
    state_path=args.output.resolve()/'execution/state.json'
    adopted_bc=None
    if args.execute and state_path.exists():
        state=json.loads(state_path.read_bytes())
        if (state.get('plan')!=binding(args.output.resolve()/'pipeline-mode.json')
                or args.execution_settings is None
                or state.get('settings')!=binding(args.execution_settings)):
            raise ValueError('existing execution bindings changed')
        completed=state.get('completed',[])
        if len(completed)>=2:
            if [r['phase'] for r in completed[:2]]!=['prepare_bc','train_bc']:
                raise ValueError('completed BC prefix differs')
            adopted_bc=completed[1]['receipt']
            bound(adopted_bc)
    if adopted_bc is None:
        immutable_json(args.output.resolve()/'bc-job.json', bc_job(p))
    result = dict(mode=args.mode, plan=binding(args.output.resolve()/'pipeline-mode.json'),
        bc_job=None if adopted_bc else binding(args.output.resolve()/'bc-job.json'),
        completed_bc_receipt=adopted_bc, GPU_work_started=False,
        requires_native_collection_and_site_execution=True, runtime_certified=False)
    if args.selected_bc_checkpoint:
        result['dagger'] = stage_dagger(p, selected_checkpoint=args.selected_bc_checkpoint,
                                      checkpoint_sha256=args.selected_bc_sha256)
    if args.execute:
        if args.execution_settings is None:parser.error('--execute requires --execution-settings')
        if args.selected_bc_checkpoint:parser.error('end-to-end execution selects its own BC parent')
        from pipeline.deployment.mode_execution import execute
        result['execution']=execute(args.output.resolve()/'pipeline-mode.json',args.execution_settings,
            allow_legacy_validation=args.allow_legacy_validation)
        result.update(GPU_work_started=True,
            runtime_certified=result['execution']['status']=='COMPLETE',
            requires_native_collection_and_site_execution=False)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
