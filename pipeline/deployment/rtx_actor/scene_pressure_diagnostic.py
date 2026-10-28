"""Isolated B16 scene-cycle diagnostic, never a production data producer.

Twenty explicit scenes per GPU exceed the failed v14 owner's 18 scene starts.
Version1 retains three teacher steps per prefix. Version2 uses ten, covering
the early captures missed before the v16 pressure failure. Both finish with
324 real steps. Neither changes production or relaxes resource admission.
This tests host pressure/camera lifecycle, not mixed-policy learning or quality.
"""
import hashlib
import json
from pathlib import Path
import time

from native_motion_assignment import validate_motion_assignment
from warm_camera_diagnostic import canonical, run_stages

SCHEMA = 'roborender.scene-pressure-diagnostic.v1'
EARLY_CAPTURE_SCHEMA = 'roborender.scene-pressure-diagnostic.v2'


def pressure_snapshot(proc=Path('/proc')):
    """Small read-only host counters; deltas diagnose stalls, not causality.

    Optional kernel counters remain absent when unsupported. No sysctl, cache
    flush, allocator change, GPU query or full process walk occurs here.
    """
    vmstat = {}
    for line in (proc / 'vmstat').read_text().splitlines():
        name, value = line.split()
        if name.startswith(('compact_', 'allocstall_', 'pgscan_', 'pgsteal_',
                            'thp_fault_', 'thp_collapse_')):
            vmstat[name] = int(value)
    pressure = {}
    for line in (proc / 'pressure/memory').read_text().splitlines():
        name, *fields = line.split()
        parsed = dict(field.split('=', 1) for field in fields)
        pressure[name] = {'total_us': int(parsed['total']),
                          'avg10': float(parsed['avg10'])}
    if set(pressure) != {'some', 'full'} or any(
            row['total_us'] < 0 or not 0 <= row['avg10'] <= 100
            for row in pressure.values()):
        raise ValueError('invalid diagnostic host pressure counters')
    return dict(monotonic_ns=time.monotonic_ns(), vmstat=vmstat,
                host_memory_pressure=pressure)


def validate_plan(value):
    if (not isinstance(value, dict) or set(value) != {'schema', 'inventory', 'arms'}
            or value['schema'] not in (SCHEMA, EARLY_CAPTURE_SCHEMA) or not isinstance(value['arms'], dict)
            or set(value['arms']) != {'0', '1'}):
        raise ValueError('exact paired scene-pressure diagnostic required')
    inventory = value['inventory']
    rows = inventory.get('motions', [])
    if (inventory.get('schema') != 'roborender.motion-inventory.v1'
            or inventory.get('motion_count') != 295 or len(rows) != 295
            or inventory.get('motion_records_sha256') != hashlib.sha256(canonical(rows)).hexdigest()):
        raise ValueError('pinned full295 inventory required')
    identities = {row['motion_id']: row for row in rows}
    if len(identities) != 295:
        raise ValueError('duplicate motion identity')
    inventory_sha = hashlib.sha256(canonical(inventory)).hexdigest()
    prefix_steps = 3 if value['schema'] == SCHEMA else 10
    for stages in value['arms'].values():
        if not isinstance(stages, list) or len(stages) != 20:
            raise ValueError('exactly twenty scenes per actor required')
        previous = None
        for index, stage in enumerate(stages):
            if not isinstance(stage, dict) or set(stage) != {'assignment', 'steps'}:
                raise ValueError('unknown scene fields')
            assignment = validate_motion_assignment(stage['assignment'])
            if (type(stage['steps']) is not int or stage['steps'] != (324 if index == 19 else prefix_steps)
                    or len(assignment['lanes']) != 16
                    or assignment['purpose'] not in {'training', 'validation'}
                    or assignment['motion_inventory_sha256'] != inventory_sha):
                raise ValueError('scene budget, B16, purpose or inventory differs')
            for lane in assignment['lanes']:
                motion = identities.get(lane['motion_id'], {})
                if (lane['sha256'] != motion.get('sha256')
                        or Path(lane['path']).name != motion.get('filename')):
                    raise ValueError('scene motion differs from inventory')
            identity = canonical(assignment)
            if identity == previous:
                raise ValueError('adjacent reused scene does not test rebuilding')
            previous = identity
        if stages[-1]['assignment']['purpose'] != 'training':
            raise ValueError('final camera soak must use training profile')
    return value


def read_plan(path, sha256):
    path = Path(path)
    if not path.is_absolute() or path.is_symlink() or not path.is_file() or path.stat().st_size > 2 << 20:
        raise ValueError('bounded regular scene-pressure plan required')
    raw = path.read_bytes()
    if len(raw) > 2 << 20 or hashlib.sha256(raw).hexdigest() != sha256:
        raise ValueError('scene-pressure plan digest differs')
    return validate_plan(json.loads(raw))


def run_native(backend, plan, gpu, report):
    # Native imports only after the owning Kit process is initialized.
    from dagger_rtx.native_entrypoints import NativeRavenRuntime, build_origin_attested_entrypoints
    from dagger_rtx.native_raven import build_native_raven_batch
    from leo_capacity_probe import run_teacher_capacity, run_teacher_camera_soak, step_acknowledgement
    from resident_isaac_trainer import reset_rollout_for_scene

    validate_plan(plan)
    trainer = backend.trainer
    stages = plan['arms'][str(gpu)]
    initial_scene_count = backend.scene_initializations

    def switch(assignment):
        backend._switch_scene('training', seed=assignment['simulator_seed'], panel=None,
                              report=report, assignment=assignment)
        trainer.env.simulator.sim.play()
        for model in (trainer.policy_model, trainer.ref_model):
            reset_rollout_for_scene(model)
        backend._prepare_training(assignment['simulator_seed'])

    def execute(steps, stage_report):
        runtime = NativeRavenRuntime.initialize(trainer)
        batch = build_native_raven_batch(trainer, build_origin_attested_entrypoints(trainer, runtime))
        probe = run_teacher_camera_soak if steps == 324 else run_teacher_capacity
        ack = ('I_ACKNOWLEDGE_324_NATIVE_CAMERA_SOAK_TEACHER_STEPS' if steps == 324
               else step_acknowledgement(steps))
        def guarded_report(row):
            trainer._native_resource_guard()
            if plan['schema'] == EARLY_CAPTURE_SCHEMA:
                row = {**row, 'host_pressure_snapshot': pressure_snapshot()}
            stage_report(row)
        result = probe(batch, trainer._rollout_lifecycle(), steps=steps,
                       acknowledgement=ack, report=guarded_report)
        trainer.env.simulator.sim.pause()
        return result

    results = run_stages(stages, switch=switch, execute=execute,
                        report=lambda row: report(row.pop('phase'), **row))
    if backend.scene_initializations - initial_scene_count != 20:
        raise RuntimeError('diagnostic did not rebuild all twenty scenes')
    steps = sum(stage['steps'] for stage in stages)
    return dict(complete=True, diagnostic_only=True, production_qualified=False,
                mixed_policy_tested=False, optimizer_steps=0, chunks_published=0,
                environment_step_calls=steps, physical_transition_rows=16*steps,
                num_envs=16, scene_rebuilds=20, stage_results=results)
