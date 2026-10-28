"""Public Isaac modes; stage protocols remain explicit, never inferred defaults.

CPU-only command normalization. This is not a capacity admission, a scene reset,
or an alternative implementation of any collection/evaluation algorithm.
"""
from copy import deepcopy
from types import MappingProxyType

SCHEMA = 'roborender.isaac-phase.v1'
ROUTES = MappingProxyType({
    'BC_COLLECT': frozenset({'collect_bc_continuous'}),
    'TEACHER_TEST': frozenset({'evaluate_scaled_teacher_panel', 'evaluate_collection_teacher'}),
    'STUDENT_TEST': frozenset({'evaluate_scaled_rendered_panel', 'evaluate_raven_pooled_rendered',
                             'evaluate_collection_mixed'}),
    'DAGGER_TRAIN': frozenset({'collect_production_segment', 'probe_continuous_training'}),
    'IDLE': frozenset({'idle'}),
})


def resolve(payload):
    """Reject wrong-policy routes before an operation is accepted or touches Isaac."""
    if (not isinstance(payload, dict)
            or set(payload) != {'schema', 'mode', 'operation', 'parameters'}
            or payload['schema'] != SCHEMA
            or not isinstance(payload['mode'], str) or payload['mode'] not in ROUTES
            or not isinstance(payload['operation'], str)
            or payload['operation'] not in ROUTES[payload['mode']]
            or not isinstance(payload['parameters'], dict)):
        raise ValueError('invalid Isaac phase selector or mode/operation mismatch')
    mode, operation, parameters = payload['mode'], payload['operation'], payload['parameters']
    if mode == 'IDLE' and parameters:
        raise ValueError('IDLE takes no parameters and cannot interrupt active work')
    if mode == 'TEACHER_TEST' and parameters.get('arm') != 'teacher':
        raise ValueError('TEACHER_TEST requires the teacher arm')
    if mode == 'STUDENT_TEST' and parameters.get('arm') not in (
            ('mixed',) if operation == 'evaluate_collection_mixed' else ('bc', 'dagger')):
        raise ValueError('STUDENT_TEST requires a frozen BC or DAgger student')
    return mode, operation, deepcopy(parameters)


def normalize_command(command):
    """Preserve outer command/worker IDs for the existing owner-bound admissions."""
    if command['operation'] != 'run_phase':
        return deepcopy(command)
    _, operation, parameters = resolve(command['payload'])
    return {**deepcopy(command), 'operation': operation, 'payload': parameters}


def mode_for(operation, payload):
    if operation == 'run_phase':
        return resolve(payload)[0]
    return 'LEGACY'  # Never relabel legacy quota/validation work as a new mode.


def validate_completion(mode, receipt):
    if not isinstance(receipt, dict) or receipt.get('complete') is not True:
        raise RuntimeError('phase did not publish complete execution evidence')
    if mode in ('TEACHER_TEST', 'STUDENT_TEST', 'BC_COLLECT'):
        if type(receipt.get('optimizer_steps')) is not int or receipt['optimizer_steps'] != 0:
            raise RuntimeError('collection/testing may not optimize the student')
    if mode in ('TEACHER_TEST', 'STUDENT_TEST') and receipt.get('purpose') != 'test':
        raise RuntimeError('test mode returned non-test evidence')
    if mode == 'DAGGER_TRAIN' and receipt.get('purpose') not in ('training', 'diagnostic_only'):
        raise RuntimeError('DAgger mode returned non-training evidence')
    if mode == 'BC_COLLECT' and receipt.get('purpose') != 'bc_collection':
        raise RuntimeError('BC mode returned non-collection evidence')
    if receipt.get('shutdown_requested'):
        raise RuntimeError('phase completion cannot shut down the warm runtime')


def capabilities(*, bc_collection=False):
    """Source routing support, explicitly NOT live qualification evidence."""
    return dict(schema=SCHEMA, native_certified=False,
        modes={mode: dict(operations=sorted(operations),
                         available=(mode != 'BC_COLLECT' or bc_collection))
               for mode, operations in ROUTES.items()},
        offline_h200_phases=['BC_RENDER', 'BC_TRAIN'],
        idle_during_offline_h200_phases=True, compute_release_supported=False)
