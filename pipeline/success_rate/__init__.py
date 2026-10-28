"""Canonical success-rate entry point with lazy historical import adapters.

Adapters contain no algorithm. Attribute access (including test monkeypatches)
forwards to the sole merged implementation. Importing the package starts no
native process and imports neither Torch nor NumPy.
"""
import importlib as _importlib
import sys as _sys
from types import ModuleType as _ModuleType


class _ModuleAlias(_ModuleType):
    def __init__(self, old, target, prefix, names):
        super().__init__(__name__ + '.' + old)
        self.__dict__.update(_target=__name__ + '.' + target, _prefix=prefix, _names=frozenset(names.split()))

    def __getattr__(self, name):
        if name in {'__path__', '__spec__'}:
            raise AttributeError(name)
        module = _importlib.import_module(self._target)
        if name == '__file__':
            return module.__file__
        if name not in self._names:
            raise AttributeError(name)
        return getattr(module, self._prefix + name)

    def __setattr__(self, name, value):
        if name in self.__dict__.get('_names', ()):
            setattr(_importlib.import_module(self._target), self._prefix + name, value)
        else:
            super().__setattr__(name, value)

    def __delattr__(self, name):
        if name in self._names:
            delattr(_importlib.import_module(self._target), self._prefix + name)
        else:
            super().__delattr__(name)

    def __dir__(self):
        return sorted(set(super().__dir__()) | self._names)


_COMPATIBILITY = {
    'production_success_contract': ('contracts', 'production_',
        'CASE_COUNT LEGACY_PANEL_ID LEGACY_PANEL_RAW_SHA256 LEGACY_PANEL_SHA256 PANEL_ID '
        'PANEL_RAW_SHA256 PANEL_SHA256 STUDENT_CLEAR_ONCE build_rendered_panel canonical '
        'control_contract evaluation_options hashlib math panel_sha256 protocol_identity '
        'require_legacy_panel require_native_timing require_panel '
        'validate_legacy_production_result validate_panel validate_production_result '
        'validate_result'
    ),
    'smoke_success_contract': ('contracts', 'smoke_',
        'PREFIX SMOKE_STUDENT_CLEAR_ONCE canonical control_contract deepcopy evaluation_options '
        'hashlib make_panel protocol_identity require_panel_smoke validate_panel '
        'validate_result validate_smoke_result'
    ),
    'rendered_student_success_contract': ('contracts', 'rendered_',
        'BINDING_SCHEMA CONFIG_SCHEMA MODE POOL_CONFIG_SCHEMAS Path SCALED_CONFIG_SCHEMA SCHEMA '
        'SHARDED_CONFIG_SCHEMA VIEWS _observation_binding _scaled_contract _sha '
        '_validate_frozen_cohort_binding _validate_scaled_receipt artifact '
        'bind_observation_config canonical configuration deepcopy digest evaluation_options '
        'hashlib make_scaled_config make_sharded_config math observation_binding panel_sha256 '
        'payload_fields protocol_identity raytraced_shape re read_inputs read_json '
        'require_panel validate_cohort_observations validate_config validate_execution '
        'validate_rendered_result validate_renderer_identity validate_result'
    ),
    'scaled_success_contract': ('contracts', 'scaled_',
        'ADMISSION_SCHEMA BATCH_SIZES CHECKPOINT_SHA256 DISTRIBUTION_SCHEMA INITIAL_ARRAYS '
        'INITIAL_SCHEMA OPERATION PARTITION_RESULT_SCHEMA PREFIX SCALED_STUDENT_CLEAR_ONCE '
        'SCHEMA _build _configuration _sha aggregate_results build_admission build_panel '
        'control_contract deepcopy digest evaluation_options explicit hashlib is_panel original '
        'protocol_identity raytraced_shape re require_panel single_partition validate_admission '
        'validate_cohort_observations validate_initial_attestation validate_result '
        'wrap_partition'
    ),
    'scaled_teacher_benchmark': ('contracts', 'teacher_',
        'AGGREGATE_SCHEMA OPERATION OWNER_FIELDS RAVEN_PPO RESULT_SCHEMA SCHEMA '
        'aggregate_results build_benchmark control_contract deepcopy digest evaluation_options '
        'explicit hashlib re require_panel reset scaled validate_benchmark validate_owner '
        'validate_parameter_attestation validate_result wrap_result'
    ),
    'episode_protocol': ('contracts', 'pooled_',
        'CLOCK_SCHEMA OPERATION OWNER_FIELDS PooledEpisodeAccounting RESULT_SCHEMA SCHEMA '
        'SOURCE_PINS build_benchmark canonical deepcopy digest hashlib json math '
        'randomize_initial_clocks re validate_benchmark validate_owner validate_result'
    ),
    'scaled_success_dispatch': ('dispatch', 'scaled_',
        'COMMIT_SCHEMA Coordinator LIMIT MAX_FINAL_RESULT_LIMIT OWNER_KEYS PLAN_SCHEMA Path '
        'Protocol Transport _owners _sha build_plan canonical deepcopy digest '
        'expected_physical_gpu fcntl hashlib immutable json os re read scaled validate_complete '
        'validate_plan validate_ready'
    ),
    'episode_dispatch': ('dispatch', 'pooled_',
        'SCHEMA aggregate benchmark build_plan deepcopy scaled validate_complete validate_plan'
    ),
    'paired_scaled_evaluation': ('runtime', 'evaluation_',
        'FAULTS FINAL_SCHEMA Path QUALIFICATION_SCHEMA RUNTIME_SCHEMA RUN_SCHEMA Run TRIALS '
        '_raw base64 bind_observation_config bound deepcopy descriptors dispatch fcntl '
        'final_panel hashlib idle_barrier json make_run math ref renderer_descriptors scaled '
        'summarize time trial_summary validate_barrier validate_config '
        'validate_dispatch_evidence validate_final_evidence validate_guard_copy '
        'validate_interval validate_qualification'
    ),
    'paired_scaled_runtime': ('runtime', '',
        'ET Path REJECT_OPTIMIZED_PYTHON REMOTE ResourceJournal Transport base64 cpu_gate '
        'deepcopy dispatch evaluate hashlib json os paired process_identity shlex subprocess '
        'time validate_runtime'
    ),
    'scaled_success_resources': ('resources', '',
        'GIB SCHEMA SUMMARY_SCHEMA _integer math memory_guard summarize'
    ),
    'scaled_success_qualification': ('resources', 'qualification_',
        'SCHEMA math request_concurrency summarize validate_qualification'
    ),
    'compare_paired_success': ('episode_metrics', 'comparison_',
        'Any Mapping PairedEvaluationError Path _evaluation _exact_mcnemar _outcomes argparse '
        'compare_reports json main math'
    ),
}
for _old, (_target, _prefix, _names) in _COMPATIBILITY.items():
    _alias = _ModuleAlias(_old, _target, _prefix, _names)
    _sys.modules[__name__ + '.' + _old] = _alias
    globals()[_old] = _alias
