"""Finalize a mode from genuine complete campaign evidence, never rerun it."""
import fcntl
import json
from pathlib import Path
import time
import uuid

from pipeline.bc_training.handoff_common import binding, bound
from pipeline.deployment.beta_operations import atomic
from pipeline.dagger.beta_campaign import immutable_json


def finalize_completed_dagger(plan_path, settings_path, *, adapter=None):
    plan_ref, settings_ref = binding(plan_path), binding(settings_path)
    plan, settings = bound(plan_ref), bound(settings_ref)
    root = Path(plan['output_root'])/'execution'
    with (root/'controller.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state_path = root/'state.json'; state = json.loads(state_path.read_bytes())
        if (state.get('status') != 'RECONCILE_REQUIRED' or state.get('phase') != 'dagger'
                or state['plan'] != plan_ref or state['settings'] != settings_ref
                or [r['phase'] for r in state['completed']] != ['prepare_bc','train_bc','test_bc']
                or settings.get('compute_release_authorized') is not False):
            raise RuntimeError('only interrupted final campaign is admissible')
        campaign_root = Path(plan['output_root'])/'dagger/campaign'
        with (campaign_root/'scheduler.lock').open('a') as campaign_lock:
            fcntl.flock(campaign_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            receipt = dict(campaign_state=binding(campaign_root/'state.json'))
            campaign = bound(receipt['campaign_state'])
            if campaign['run_id'] != 'dagger-leo-'+plan['run_id'] or campaign.get('phase') != 'COMPLETE':
                raise RuntimeError('exact complete campaign required')
            if adapter is None:
                from pipeline.deployment.mode_site import ModeSite
                adapter = ModeSite(plan, settings)
            for row in state['completed']:
                adapter.validate(row['phase'], bound(row['receipt']), state['completed'])
            adapter.validate('dagger', receipt, state['completed'])
            output = root/'dagger-receipt.json'
            if output.exists():
                raise RuntimeError('receipt already exists; reconcile it without overwrite')
            archive = root/'recoveries'/uuid.uuid4().hex
            immutable_json(archive/'prior-state.json', state)
            immutable_json(archive/'admission.json', dict(schema='roborender.completed-mode-recovery.v1',
                campaign_state=receipt['campaign_state'], native_commands_replayed=0,
                optimizer_steps_replayed=0, compute_release_authorized=False))
            atomic(output, receipt)
            state['completed'].append(dict(phase='dagger', receipt=binding(output),
                wall_seconds=time.time()-state['started_at']))
            state.update(status='COMPLETE', completed_campaign_recovery=str(archive/'admission.json'))
            state.pop('error_type', None)
            atomic(state_path, state)
            return state
