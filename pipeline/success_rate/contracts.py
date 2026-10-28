"""Consolidated success evaluation contracts; versioned protocols remain distinct."""
from __future__ import annotations


# ---- collection-bound equal-repeat student phase (CPU plan, not admission) ----

COLLECTION_REFERENCE_SCHEMA = 'roborender.bc-initial-motion-reference.v1'
COLLECTION_STUDENT_SCHEMA = 'roborender.collection-bound-student-test-plan.v1'
INITIAL_STUDENT_SHA256 = '720a975812f6083192234d650033511c821b6902da2ef7734d87943f6c0a359a'


def collection_initial_reference(request_raw):
    """Preserve original request bytes and ordered lane support, not fake poses.

    Fresh repeats draw reset states from these motion conditions. The original
    request is retained verbatim so this catalog does not depend on a run path
    remaining available. Native physical/RNG snapshots are not reconstructed.
    """
    import hashlib
    import json
    from copy import deepcopy
    from pipeline.deployment.rtx_actor.isaac6_bc_production import validate_collection_request
    from pipeline.success_rate.teacher_episodes import digest
    if not isinstance(request_raw, bytes) or not 0 < len(request_raw) <= 1 << 20:
        raise ValueError('bounded original BC request bytes required')
    request = validate_collection_request(json.loads(request_raw))
    motions, inventory = request['motions'], request['motion_inventory']
    if (len(motions) != 32 or motions != inventory[:32]
            or request.get('global_env_offset', 0) != 0):
        raise ValueError('exact ordered first32 BC motion assignments required')
    return dict(schema=COLLECTION_REFERENCE_SCHEMA, bc_run_id=request['run_id'],
        source_request_sha256=hashlib.sha256(request_raw).hexdigest(),
        source_request_utf8=request_raw.decode('utf-8'),
        collection_seed=request['process_seed'], num_conditions=32,
        lane_assignments=[dict(lane=i, motion=deepcopy(m)) for i, m in enumerate(motions)],
        ordered_motion_sha256=digest(motions), full_reset_inventory_sha256=digest(inventory),
        full_reset_inventory_count=len(inventory),
        semantics='motion_support_and_reset_inputs_not_historical_simulator_snapshot',
        exact_historical_state_available=False,
        evaluation_sampling='fresh_native_resets_from_same_ordered_motion_conditions')


def validate_collection_initial_reference(value):
    from copy import deepcopy
    if not isinstance(value, dict) or not isinstance(value.get('source_request_utf8'), str):
        raise ValueError('BC initial-condition reference requires original request bytes')
    expected = collection_initial_reference(value['source_request_utf8'].encode('utf-8'))
    if value != expected:
        raise ValueError('BC initial-condition reference differs from original collection request')
    return deepcopy(value)


def collection_student_plan(reference, *, checkpoint_sha256, renderer, seeds):
    """Explicit untrained-student smoke definition; never authorizes execution.

    Renderer selection is independent of checkpoint lineage. In particular,
    offline teacher50 is not an implemented causal online renderer merely
    because the checkpoint is a BC checkpoint.
    """
    from pipeline.success_rate.teacher_episodes import digest
    reference = validate_collection_initial_reference(reference)
    if checkpoint_sha256 != INITIAL_STUDENT_SHA256:
        raise ValueError('this diagnostic requires the preserved initial student, not a trained/retired parent')
    if (not isinstance(seeds, (tuple, list)) or len(seeds) != 2
            or any(type(s) is not int or not 0 <= s < 2**31 for s in seeds)
            or len(set(seeds)) != 2 or reference['collection_seed'] in seeds):
        raise ValueError('two distinct fresh actor reset seeds required')
    profiles = {
        'online_image3f5': dict(adapter_sha256='c68de259ce5ee8e4df5a45296be8328242b3ebba351990df1d86f4e3a7a2d588',
            diffusion_steps=5, cfg=1.0, slot_order=['head_color', 'hand_left', 'hand_right'],
            role='causal_online', reference='fixed_frame0_per_episode',
            causal_adapter_implemented=True),
        'offline_bc_teacher50': dict(adapter_sha256='3f09aa23a26ddb2da32d027dc5e883c992c820854bc07020d6d254c806afb01b',
            diffusion_steps=50, cfg=5.0, slot_order=['hand_left', 'hand_right', 'head_color'],
            role='offline_clip_renderer_requires_separate_online_adapter',
            reference='separately_versioned_causal_contract_required',
            causal_adapter_implemented=False),
    }
    if renderer not in profiles:
        raise ValueError('explicit named renderer required; no renderer1/2 or native fallback')
    return dict(schema=COLLECTION_STUDENT_SCHEMA, mode='STUDENT_TEST', purpose='diagnostic',
        initial_reference_sha256=digest(reference), checkpoint_sha256=checkpoint_sha256,
        policy='initial_untrained_student', optimizer_steps=0, teacher_action_probability=0,
        teacher_takeover=False, student_action='deterministic_mean11_plus_homie15',
        renderer=dict(name=renderer, **profiles[renderer]), generated_only=True,
        raytraced_fallback_allowed=False, policy_view_order=['head_color', 'hand_left', 'hand_right'],
        rendering_input='current_synchronized_depth_and_textured_robot_rgb_three_views',
        all_lane_render_action_barrier=True, partial_batch_step_allowed=False,
        future_state_conditioning_allowed=False,
        actors=[dict(physical_gpu=i, num_envs=32, reset_seed=seeds[i],
                     motion_assignment='same_ordered32_on_both_actors',
                     h200_slots=list(range(i*4, i*4+4))) for i in (0, 1)],
        distinct_motion_conditions=32, concurrent_environments=64,
        episodes_per_lane=3, expected_per_actor=96, expected_total=192,
        initial_clocks='native_reset_episode0_stages1_full_episode_budget',
        natural_done_timeout_resets_only=True, chunk_count_reset_allowed=False,
        extra_fast_lane_episodes='unscored_not_reset_or_paused_at_quota',
        success='native_pre_reset_goal_OR_until_natural_done',
        aggregation='sum_successes_divided_by_sum_scored_completions',
        incomplete_safety_stop='partial_not_success_or_failure',
        final_state='same_Kit_physics_idle_render_sessions_closed',
        checkpoint_and_running_statistics_frozen=True,
        native_replay_allowed=False, compute_release_authorized=False,
        ready=False, execution_admission=None,
        pending=['collection_bound_student_native_adapter', 'source_bound_owner_admission',
                 'actual_renderer_checkpoint_identity_and_paired_B32_capacity'])


def validate_collection_student_plan(value, reference):
    if not isinstance(value, dict):
        raise ValueError('student plan must be an object')
    expected = collection_student_plan(reference, checkpoint_sha256=value.get('checkpoint_sha256'),
        renderer=value.get('renderer', {}).get('name'),
        seeds=[actor.get('reset_seed') for actor in value.get('actors', [])])
    if value != expected:
        raise ValueError('student phase plan differs from collection-bound diagnostic contract')
    return expected


# ---- production success contract ----

import hashlib as production_hashlib
import math as production_math

from pipeline.deployment.rtx_actor.explicit_motion_panel import canonical as production_canonical, panel_sha256 as production_panel_sha256, validate_panel as production_validate_panel, validate_result as production_validate_result
from pipeline.deployment.rtx_actor.initial_robot_refresh_diagnostic import STUDENT_CLEAR_ONCE as production_STUDENT_CLEAR_ONCE, control_contract as production_control_contract

production_LEGACY_PANEL_RAW_SHA256 = "56f5b2205a840558db477ef75a3ff6e62962af107b6b372ab23320970424ee80"
production_LEGACY_PANEL_SHA256 = "627b60f1b82677ebe168c947836b03e7e2eabda24e67e250a32103efc7bab377"
production_LEGACY_PANEL_ID = "plan-only-native295-b2-20261015-grouped-test"
production_PANEL_ID = "explicit-production-image3f-fixed295-v1"
production_PANEL_RAW_SHA256 = "5bae2865505a85b5c186df668654ad14a7df42ec81371d660cf3a67c83e6013c"
production_PANEL_SHA256 = "b9dea9ddfaf5e0196a225c472abd956886afaa8cc4ea17a43372a07eb6d5d184"
production_CASE_COUNT = 295


def production_require_legacy_panel(panel, *, raw_sha256=None):
    """Read-only historical panel admission; never an execution option."""
    value = production_validate_panel(panel)
    if (value['observation_mode'] != 'raytraced_3view' or production_panel_sha256(value) != production_LEGACY_PANEL_SHA256
            or (raw_sha256 is not None and raw_sha256 != production_LEGACY_PANEL_RAW_SHA256)):
        raise ValueError('historical fixed295 source panel differs')
    return value


def production_build_rendered_panel(legacy_panel):
    """Change only version label/visual domain; keep every reset assignment."""
    value = production_require_legacy_panel(legacy_panel)
    value.update(panel_id=production_PANEL_ID, observation_mode='image3f_3view')
    return production_require_panel(value)


def production_validate_legacy_production_result(panel, result, checkpoint_sha256):
    """Read historical ray-traced receipts only; never authorize execution.

    Keeping this explicit reader avoids rewriting old evidence as Image3F.
    Every execution admission still goes through require_panel/evaluation_options.
    """
    production_require_legacy_panel(panel)
    identity = production_protocol_identity(production_build_rendered_panel(panel))
    for key in ('sha256', 'raytraced_fallback_allowed', 'legacy_case_assignment_panel_sha256'):
        identity.pop(key)
    identity.update(schema='roborender.production-student-success-protocol.v1',
        panel_raw_sha256=production_LEGACY_PANEL_RAW_SHA256, panel_sha256=production_LEGACY_PANEL_SHA256,
        observation_mode='raytraced_3view')
    identity['sha256'] = production_hashlib.sha256(production_canonical(identity)).hexdigest()
    if not isinstance(result, dict) or result.get('production_success_protocol') != identity:
        raise ValueError('historical ray-traced protocol receipt differs')
    return production_validate_result(panel, result, checkpoint_sha256, allowed_arms=('bc', 'dagger'),
        teacher_action_mode='distill', diagnostic_initial_robot_refresh=production_STUDENT_CLEAR_ONCE)


def production_require_panel(panel, *, raw_sha256=None):
    value = production_validate_panel(panel)
    if (value['observation_mode'] != 'image3f_3view' or value['panel_id'] != production_PANEL_ID
            or value["purpose"] != "test" or value["episode_count"] != production_CASE_COUNT
            or production_panel_sha256(value) != production_PANEL_SHA256
            or (raw_sha256 is not None and raw_sha256 != production_PANEL_RAW_SHA256)
            or [len(c["case_ids"]) for c in value["cohorts"]] != [2] * 147 + [1]):
        raise ValueError("production success requires the exact rendered295 panel; ray-traced student execution is forbidden")
    return value


def production_evaluation_options(panel, *, raw_sha256=None):
    production_require_panel(panel, raw_sha256=raw_sha256)
    return {"teacher_action_mode": "distill",
            "diagnostic_initial_robot_refresh": production_STUDENT_CLEAR_ONCE}


def production_protocol_identity(panel):
    production_require_panel(panel)
    identity = {
        "schema": "roborender.production-student-success-protocol.v2",
        "panel_raw_sha256": production_PANEL_RAW_SHA256, "panel_sha256": production_PANEL_SHA256,
        "completed_episodes": production_CASE_COUNT, "cohort_sizes": [2] * 147 + [1],
        "observation_mode": "image3f_3view", "max_episode_seconds": 8,
        "raytraced_fallback_allowed": False,
        "legacy_case_assignment_panel_sha256": production_LEGACY_PANEL_SHA256,
        "control_hz": 50, "physics_hz": 200,
        "metric": "native_goal_or_until_done",
        "goal_accounting_version": "roborender.raven.pre_reset_goal.v1",
        "behavior": "deterministic_student_action_mean_11_plus_homie_15",
        "teacher_action_probability": 0, "optimizer_steps": 0,
        "reset_control": production_control_contract(production_STUDENT_CLEAR_ONCE),
    }
    return {**identity, "sha256": production_hashlib.sha256(production_canonical(identity)).hexdigest()}


def production_validate_production_result(panel, result, checkpoint_sha256, *, allowed_arms=("bc", "dagger"), rendered_config=None):
    options = production_evaluation_options(panel)
    if not allowed_arms or not set(allowed_arms) <= {"bc", "dagger"}:
        raise ValueError("production success must evaluate a student, never teacher takeover")
    if not isinstance(result, dict) or result.get("production_success_protocol") != production_protocol_identity(panel):
        raise ValueError("production success result lacks the corrected fixed-panel protocol identity")
    production_validate_result(panel, result, checkpoint_sha256, allowed_arms=allowed_arms, **options)
    from pipeline.success_rate.rendered_student_success_contract import validate_rendered_result
    return validate_rendered_result(panel, result, checkpoint_sha256, rendered_config)


def production_require_native_timing(env):
    """Read actual initialized timing; do not mutate physics or the horizon."""
    values = {"max_episode_seconds": float(env.max_episode_length_s),
              "control_dt": float(env.dt), "physics_dt": float(env.sim_dt),
              "max_episode_control_steps": float(env.max_episode_length)}
    expected = {"max_episode_seconds": 8.0, "control_dt": 0.02,
                "physics_dt": 0.005, "max_episode_control_steps": 400.0}
    if any(not production_math.isfinite(values[k]) or not production_math.isclose(values[k], v, rel_tol=0, abs_tol=1e-10)
           for k, v in expected.items()):
        raise ValueError("production success native horizon/control/physics timing differs")
    return values


# ---- smoke success contract ----
from copy import deepcopy as smoke_deepcopy
import hashlib as smoke_hashlib

from pipeline.deployment.rtx_actor.explicit_motion_panel import canonical as smoke_canonical, validate_panel as smoke_validate_panel, validate_result as smoke_validate_result
from pipeline.deployment.rtx_actor.initial_robot_refresh_diagnostic import SMOKE_STUDENT_CLEAR_ONCE as smoke_SMOKE_STUDENT_CLEAR_ONCE, control_contract as smoke_control_contract

smoke_PREFIX = "explicit-smoke-image3f-corrected-"


def smoke_make_panel(accepted_panel):
    from pipeline.success_rate.production_success_contract import require_panel, PANEL_SHA256
    parent = require_panel(accepted_panel)
    panel = smoke_deepcopy(parent)
    panel.update(panel_id=smoke_PREFIX + PANEL_SHA256[:16], episode_count=2,
                 cohorts=smoke_deepcopy(parent['cohorts'][:1]))
    panel['cohorts'][0]['assignment']['run_id'] = panel['panel_id']
    return smoke_require_panel_smoke(panel)


def smoke_require_panel_smoke(panel):
    value = smoke_validate_panel(panel)
    if (not value['panel_id'].startswith(smoke_PREFIX) or value['observation_mode'] != 'image3f_3view' or value['purpose'] != 'test'
            or value['episode_count'] != 2 or len(value['cohorts']) != 1
            or len(value['cohorts'][0]['assignment']['lanes']) != 2):
        raise ValueError('corrected smoke requires its explicitly named two-case panel')
    return value


def smoke_evaluation_options(panel):
    smoke_require_panel_smoke(panel)
    return dict(teacher_action_mode='distill',
                diagnostic_initial_robot_refresh=smoke_SMOKE_STUDENT_CLEAR_ONCE)


def smoke_protocol_identity(panel):
    panel = smoke_require_panel_smoke(panel)
    value = dict(schema='roborender.smoke-student-success-protocol.v2',
        panel_sha256=smoke_hashlib.sha256(smoke_canonical(panel)).hexdigest(), completed_episodes=2,
        optimizer_steps=0, production_qualified=False,
        observation_mode='image3f_3view', raytraced_fallback_allowed=False, max_episode_seconds=8,
        control_hz=50, physics_hz=200, metric='native_goal_or_until_done',
        reset_control=smoke_control_contract(smoke_SMOKE_STUDENT_CLEAR_ONCE))
    return {**value, 'sha256': smoke_hashlib.sha256(smoke_canonical(value)).hexdigest()}


def smoke_validate_smoke_result(panel, result, checkpoint_sha256, *, allowed_arms=('bc', 'dagger'), rendered_config=None):
    if result.get('smoke_success_protocol') != smoke_protocol_identity(panel):
        raise ValueError('smoke result is missing its corrected reset/panel identity')
    smoke_validate_result(panel, result, checkpoint_sha256, allowed_arms=allowed_arms,
                    **smoke_evaluation_options(panel))
    from pipeline.success_rate.rendered_student_success_contract import validate_rendered_result
    return validate_rendered_result(panel, result, checkpoint_sha256, rendered_config)


# ---- rendered student success contract ----
from copy import deepcopy as rendered_deepcopy
import hashlib as rendered_hashlib
import math as rendered_math
from pathlib import Path as rendered_Path
import re as rendered_re

from pipeline.deployment.rtx_actor.explicit_motion_panel import canonical as rendered_canonical, panel_sha256 as rendered_panel_sha256, validate_result as rendered_validate_result
from pipeline.deployment.rtx_actor.rendered_training_support_contract import validate_renderer_identity as rendered_validate_renderer_identity, read_json as rendered_read_json

rendered_SCHEMA = 'roborender.rendered-student-success-protocol.v1'
rendered_CONFIG_SCHEMA = 'roborender.rendered-student-success-config.v1'
rendered_SCALED_CONFIG_SCHEMA = 'roborender.rendered-student-success-config.v2'
rendered_SHARDED_CONFIG_SCHEMA = 'roborender.rendered-student-success-config.v3'
rendered_POOL_CONFIG_SCHEMAS = {rendered_SCALED_CONFIG_SCHEMA, rendered_SHARDED_CONFIG_SCHEMA}
rendered_BINDING_SCHEMA = 'roborender.rendered-student-success-cohort-binding.v1'
rendered_MODE = 'image3f_3view'
rendered_VIEWS = ['head_color', 'hand_left', 'hand_right']


def rendered_digest(value):
    return rendered_hashlib.sha256(rendered_canonical(value)).hexdigest()


def rendered__sha(value):
    return isinstance(value, str) and rendered_re.fullmatch(r'[0-9a-f]{64}', value) is not None


def rendered__scaled_contract(panel):
    # Import only for the explicit opt-in version; historical/v1 imports and
    # validation do not depend on a scaled panel being installed.
    if isinstance(panel, dict) and str(panel.get('panel_id', '')).startswith('scaled-rendered295-'):
        from pipeline.success_rate import scaled_success_contract
        return scaled_success_contract
    return None


def rendered_raytraced_shape(panel):
    """Structural legacy parser only; never an executable visual fallback."""
    if scaled := rendered__scaled_contract(panel):
        return scaled.raytraced_shape(panel)
    from pipeline.success_rate.production_success_contract import PANEL_ID, LEGACY_PANEL_ID
    from pipeline.success_rate.smoke_success_contract import PREFIX
    if not isinstance(panel, dict) or panel.get('observation_mode') != rendered_MODE:
        raise ValueError('rendered student panel required')
    value = rendered_deepcopy(panel)
    if value.get('panel_id') == PANEL_ID:
        value['panel_id'] = LEGACY_PANEL_ID
    elif not isinstance(value.get('panel_id'), str) or not value['panel_id'].startswith(PREFIX):
        raise ValueError('unknown rendered student panel version')
    value['observation_mode'] = 'raytraced_3view'
    return value


def rendered_require_panel(panel):
    if scaled := rendered__scaled_contract(panel):
        return scaled.require_panel(panel)
    from pipeline.success_rate.production_success_contract import require_panel as production, PANEL_ID
    from pipeline.success_rate.smoke_success_contract import require_panel_smoke
    if isinstance(panel, dict) and panel.get('panel_id') == PANEL_ID:
        return production(panel)
    return require_panel_smoke(panel)


def rendered_evaluation_options(panel):
    if scaled := rendered__scaled_contract(panel):
        return scaled.evaluation_options(panel)
    from pipeline.success_rate.production_success_contract import evaluation_options as production, PANEL_ID
    from pipeline.success_rate.smoke_success_contract import evaluation_options as smoke
    rendered_require_panel(panel)
    return production(panel) if panel['panel_id'] == PANEL_ID else smoke(panel)


def rendered_validate_config(config, panel):
    rendered_require_panel(panel)
    if (not isinstance(config, dict) or set(config) != {'schema', 'checkpoint_sha256', 'panel_sha256', 'renderer', 'anchors'}
            or config['schema'] not in {rendered_CONFIG_SCHEMA, *rendered_POOL_CONFIG_SCHEMAS} or not rendered__sha(config['checkpoint_sha256'])
            or config['panel_sha256'] != rendered_panel_sha256(panel)):
        raise ValueError('rendered student configuration must bind exact checkpoint and panel')
    renderer, anchors = config['renderer'], config['anchors']
    scaled = rendered__scaled_contract(panel)
    if config['schema'] in rendered_POOL_CONFIG_SCHEMAS:
        if scaled is None:
            raise ValueError('scaled endpoint configuration requires an explicit scaled panel')
        from pipeline.deployment.rtx_actor.rendered_endpoint_pool import validate_pool_config, SHARDED_ROUTING
        validate_pool_config(renderer)
        if (config['schema'] == rendered_SHARDED_CONFIG_SCHEMA) != (renderer['routing'] == SHARDED_ROUTING):
            raise ValueError('eight-renderer actor shards require the explicit v3 configuration')
    elif (scaled is not None or not isinstance(renderer, dict) or set(renderer) != {'host', 'port', 'identity'}
            or renderer['host'] != '127.0.0.1' or type(renderer['port']) is not int
            or not 1024 <= renderer['port'] <= 65535):
        raise ValueError('v1 requires the unchanged single loopback renderer and original panel')
    if (not isinstance(anchors, dict) or set(anchors) != {'seed', 'files'}
            or type(anchors['seed']) is not int or not 0 <= anchors['seed'] < 2 ** 32
            or not isinstance(anchors['files'], list) or len(anchors['files']) != 2048):
        raise ValueError('loopback renderer and independently seeded full2048 anchor inventory required')
    rendered_validate_renderer_identity(renderer['identity'])
    for row in anchors['files']:
        if (not isinstance(row, dict) or set(row) != {'path', 'sha256'}
                or not isinstance(row['path'], str) or not rendered_Path(row['path']).is_absolute()
                or '..' in rendered_Path(row['path']).parts or not rendered__sha(row['sha256'])):
            raise ValueError('exact hash-bound absolute anchor path required')
    if len({row['path'] for row in anchors['files']}) != 2048:
        raise ValueError('duplicate anchor inventory path')
    rendered_canonical(config)
    return rendered_deepcopy(config)


def rendered_bind_observation_config(template, panel, checkpoint_sha256):
    """Bind shared renderer assets anew for each selected stage checkpoint."""
    if not isinstance(template, dict) or not {'renderer', 'anchors'} <= set(template):
        raise ValueError('rendering dependency template required')
    from pipeline.deployment.rtx_actor.rendered_endpoint_pool import SHARDED_ROUTING
    schema = ((rendered_SHARDED_CONFIG_SCHEMA if template['renderer'].get('routing') == SHARDED_ROUTING else rendered_SCALED_CONFIG_SCHEMA)
              if rendered__scaled_contract(panel) else rendered_CONFIG_SCHEMA)
    return rendered_validate_config(dict(schema=schema, checkpoint_sha256=checkpoint_sha256,
        panel_sha256=rendered_panel_sha256(rendered_require_panel(panel)), renderer=rendered_deepcopy(template['renderer']),
        anchors=rendered_deepcopy(template['anchors'])), panel)


def rendered_make_scaled_config(template, panel, checkpoint_sha256, endpoints):
    """Build the opt-in config from actually admitted endpoint receipt pins.

    This is a pure builder, not GPU/source qualification. The caller must check
    each raw readiness receipt and owner/GPU/source before native admission;
    the adapter independently checks its live generation source and process.
    A single endpoint permits a component probe, not four-GPU qualification.
    """
    if rendered__scaled_contract(panel) is None:
        raise ValueError('scaled configuration requires a versioned scaled panel')
    if not isinstance(template, dict) or not {'renderer', 'anchors'} <= set(template):
        raise ValueError('rendering dependency template required')
    renderer = dict(identity=rendered_deepcopy(template['renderer']['identity']), endpoints=rendered_deepcopy(endpoints),
        routing='lane_mod_endpoint_count', max_inflight_per_endpoint=1)
    return rendered_validate_config(dict(schema=rendered_SCALED_CONFIG_SCHEMA, checkpoint_sha256=checkpoint_sha256,
        panel_sha256=rendered_panel_sha256(rendered_require_panel(panel)), renderer=renderer,
        anchors=rendered_deepcopy(template['anchors'])), panel)


def rendered_make_sharded_config(template, panel, checkpoint_sha256, endpoints):
    """Two RTX actors, four exclusive batch-one renderer engines per actor."""
    from pipeline.deployment.rtx_actor.rendered_endpoint_pool import ACTOR_SHARDS, SHARDED_ROUTING
    if rendered__scaled_contract(panel) is None or not isinstance(template, dict) or not {'renderer', 'anchors'} <= set(template):
        raise ValueError('sharded rendering requires explicit scaled panel and dependency template')
    renderer = dict(identity=rendered_deepcopy(template['renderer']['identity']), endpoints=rendered_deepcopy(endpoints),
        routing=SHARDED_ROUTING, max_inflight_per_endpoint=1, actor_shards=rendered_deepcopy(ACTOR_SHARDS))
    return rendered_validate_config(dict(schema=rendered_SHARDED_CONFIG_SCHEMA, checkpoint_sha256=checkpoint_sha256,
        panel_sha256=rendered_panel_sha256(rendered_require_panel(panel)), renderer=renderer,
        anchors=rendered_deepcopy(template['anchors'])), panel)


def rendered_read_inputs(config_path, raw_sha256, panel, *, checkpoint_sha256=None):
    config = rendered_validate_config(rendered_read_json(config_path, raw_sha256), panel)
    if checkpoint_sha256 is not None and config['checkpoint_sha256'] != checkpoint_sha256:
        raise ValueError('rendering configuration checkpoint differs')
    return config


def rendered_validate_execution(panel, checkpoint_sha256, config, *, arm='bc', has_resume=False):
    config = rendered_validate_config(config, panel)
    if arm not in {'bc', 'dagger'} or checkpoint_sha256 != config['checkpoint_sha256']:
        raise ValueError('rendered evaluation requires the exact selected student checkpoint')
    if type(has_resume) is not bool:
        raise ValueError('explicit completed-cohort resume state required')
    return config


def rendered_observation_binding(panel, config, cohort_index):
    config = rendered_validate_config(config, panel)
    if type(cohort_index) is not int or not 0 <= cohort_index < len(panel['cohorts']):
        raise ValueError('rendered cohort index differs')
    return rendered__observation_binding(panel, config, cohort_index)


def rendered__observation_binding(panel, config, cohort_index):
    """Internal builder after the public boundary validated the frozen inputs."""
    return dict(schema=rendered_BINDING_SCHEMA, checkpoint_sha256=config['checkpoint_sha256'],
        panel_sha256=config['panel_sha256'], config_sha256=rendered_digest(config), cohort_index=cohort_index,
        motion_assignment_sha256=rendered_digest(panel['cohorts'][cohort_index]['assignment']))


def rendered_validate_cohort_observations(result, *, count, config=None, cohort_index=None, panel=None, _pooled_benchmark=None):
    max_steps = 450
    if _pooled_benchmark is not None:
        from pipeline.success_rate.episode_protocol import validate_benchmark, validate_result
        validate_benchmark(_pooled_benchmark, panel, config)
        validate_result(result, _pooled_benchmark)
        max_steps = _pooled_benchmark['max_vector_steps']
    receipt = result.get('rendered_observation_receipt')
    steps = result.get('vector_steps')
    expected = dict(schema='roborender.rendered-success-observation-receipt.v1', observation_mode=rendered_MODE,
        view_order=rendered_VIEWS, policy_view_shape=[108, 192, 3], control_view_shape=[240, 416],
        raytraced_fallbacks=0, failed=False, normalizations_per_vector_step=1,
        anchor_policy='fixed_frame0_per_episode', controls_from_current_simulator_state=True,
        simulator_steps_added=0, optimizer_steps=0, offline_BC_teacher50_equivalent=False)
    if (not isinstance(receipt, dict) or type(steps) is not int or not 1 <= steps <= max_steps
            or type(count) is not int or not 1 <= count <= 40
            or any(receipt.get(k) != v for k, v in expected.items())
            or receipt.get('vector_steps_rendered') != steps or receipt.get('render_requests_completed') != count * steps
            or not rendered__sha(receipt.get('trace_sha256')) or type(receipt.get('anchor_seed')) is not int
            or not 0 <= receipt['anchor_seed'] < 2 ** 32
            or type(receipt.get('elapsed_seconds')) not in (int, float)
            or not rendered_math.isfinite(receipt['elapsed_seconds']) or receipt['elapsed_seconds'] < 0
            or 'paired_initial_state_attestation' in result):
        raise ValueError('rendered student cohort lacks exact all-lane observation evidence')
    rendered_validate_renderer_identity(receipt.get('renderer_identity'))
    binding = receipt.get('evaluation_binding')
    if (not isinstance(binding, dict) or set(binding) != {'schema', 'checkpoint_sha256', 'panel_sha256',
            'config_sha256', 'cohort_index', 'motion_assignment_sha256'} or binding['schema'] != rendered_BINDING_SCHEMA
            or any(not rendered__sha(binding[k]) for k in ('checkpoint_sha256', 'panel_sha256', 'config_sha256', 'motion_assignment_sha256'))
            or type(binding['cohort_index']) is not int or not 0 <= binding['cohort_index'] < 300):
        raise ValueError('rendered observation missing per-cohort configuration binding')
    for field in ('checkpoint_sha256', 'motion_assignment_sha256'):
        if field in result and result[field] != binding[field]:
            raise ValueError('rendered binding differs from native cohort')
    anchors = receipt.get('anchor_selections')
    if not isinstance(anchors, list) or not count <= len(anchors) <= count * steps:
        raise ValueError('rendered anchor selection coverage differs')
    seen = set()
    for row in anchors:
        keys = {'lane', 'selected_at_vector_step', 'file_sha256', 'frame_index', 'prompt_sha256'}
        if _pooled_benchmark is not None:
            keys |= {'episode_index', 'episode_session_id'}
        if (not isinstance(row, dict) or set(row) != keys
                or type(row['lane']) is not int or not 0 <= row['lane'] < count
                or type(row['selected_at_vector_step']) is not int or not 0 <= row['selected_at_vector_step'] < steps
                or row['frame_index'] != 0 or not rendered__sha(row['file_sha256']) or not rendered__sha(row['prompt_sha256'])
                or (row['lane'], row['selected_at_vector_step']) in seen):
            raise ValueError('invalid fixed-per-episode anchor receipt')
        seen.add((row['lane'], row['selected_at_vector_step']))
    if {(lane, 0) for lane in range(count)} - seen:
        raise ValueError('missing initial anchor for a physical lane')
    inputs = receipt.get('bounded_first_inputs')
    if not isinstance(inputs, list) or len(inputs) != min(2, steps):
        raise ValueError('missing bounded renderer-input digests')
    for step, record in enumerate(inputs):
        if (not isinstance(record, dict) or set(record) != {'step', 'lanes'} or record['step'] != step
                or not isinstance(record['lanes'], list) or len(record['lanes']) != count):
            raise ValueError('renderer-input lane/step digest coverage differs')
        for lane, entry in enumerate(record['lanes']):
            if (not isinstance(entry, dict) or set(entry) != {'lane', 'inputs', 'generated'} or entry['lane'] != lane
                    or not isinstance(entry['inputs'], dict) or set(entry['inputs']) != {'depth', 'mask', 'previous'}):
                raise ValueError('renderer-input digest keys differ')
            for kind, views in [*entry['inputs'].items(), ('generated', entry['generated'])]:
                if not isinstance(views, dict) or set(views) != set(rendered_VIEWS):
                    raise ValueError('renderer digest lacks all three views')
                for descriptor in views.values():
                    if (not isinstance(descriptor, dict) or set(descriptor) != {'shape', 'dtype', 'sha256'} or descriptor['dtype'] != 'uint8'
                            or descriptor['shape'] != ([240, 416] if kind == 'depth' else [240, 416, 3])
                            or not rendered__sha(descriptor['sha256'])):
                        raise ValueError('renderer input/output geometry or digest differs')
    if config is not None:
        config = rendered_validate_config(config, panel)
        if type(cohort_index) is not int or not 0 <= cohort_index < len(panel['cohorts']):
            raise ValueError('rendered cohort index differs')
        rendered__validate_frozen_cohort_binding(receipt, panel, config, cohort_index,
            {a['sha256'] for a in config['anchors']['files']})
    # Scaled receipts must remain structurally self-validating when the panel
    # summarizer checks them before the frozen config is supplied.
    if 'renderer_endpoint_pool' in receipt or 'phase_timing' in receipt:
        rendered__validate_scaled_receipt(receipt, count=count, steps=steps,
            renderer=config['renderer'] if config and config['schema'] in rendered_POOL_CONFIG_SCHEMAS else None)
    return rendered_deepcopy(receipt)


def rendered__validate_scaled_receipt(receipt, *, count, steps, renderer=None):
    from pipeline.deployment.rtx_actor.rendered_endpoint_pool import validate_pool_receipt, validate_timing_receipt
    validate_pool_receipt(receipt.get('renderer_endpoint_pool'), count=count, steps=steps, renderer=renderer,
        cohort_index=receipt.get('evaluation_binding', {}).get('cohort_index'))
    validate_timing_receipt(receipt.get('phase_timing'), steps=steps)
    transports = receipt.get('renderer_transport_receipts')
    bindings = receipt['renderer_endpoint_pool']['lane_bindings']
    if not isinstance(transports, list) or len(transports) != count:
        raise ValueError('scaled observation lacks every pinned transport completion')
    for row, binding in zip(transports, bindings):
        if (not isinstance(row, dict) or row.get('schema') != 'roborender.image3f.cached-reply.v1'
                or row.get('server_instance') != binding['server_instance']
                or row.get('channel') != binding['channel'] or row.get('failed') is not False
                or type(row.get('logical_requests_completed')) is not int
                or row['logical_requests_completed'] != steps
                or type(row.get('transport_attempts')) is not int or row['transport_attempts'] < steps
                or type(row.get('cached_replies_recovered')) is not int
                or not 0 <= row['cached_replies_recovered'] <= steps):
            raise ValueError('scaled transport completion differs from pinned lane/worker sequence')


def rendered__validate_frozen_cohort_binding(receipt, panel, config, cohort_index, allowed_hashes):
    if (receipt['evaluation_binding'] != rendered__observation_binding(panel, config, cohort_index)
            or receipt['renderer_identity'] != config['renderer']['identity']
            or receipt['anchor_seed'] != (config['anchors']['seed'] + cohort_index) % (2 ** 32)
            or any(row['file_sha256'] not in allowed_hashes for row in receipt['anchor_selections'])):
        raise ValueError('rendered cohort differs from frozen configuration')
    if config['schema'] in rendered_POOL_CONFIG_SCHEMAS:
        rendered__validate_scaled_receipt(receipt, count=len(panel['cohorts'][cohort_index]['case_ids']),
            steps=receipt['vector_steps_rendered'], renderer=config['renderer'])


def rendered_protocol_identity(panel, config):
    config = rendered_validate_config(config, panel)
    value = dict(schema=rendered_SCHEMA, observation_mode=rendered_MODE, panel_sha256=rendered_panel_sha256(panel),
        checkpoint_sha256=config['checkpoint_sha256'], configuration_sha256=rendered_digest(config),
        renderer_identity=rendered_deepcopy(config['renderer']['identity']), anchor_inventory_sha256=rendered_digest(config['anchors']['files']),
        anchor_seed=config['anchors']['seed'], cohort_anchor_seed_rule='(anchor_seed + original_cohort_index) mod 2**32; independent RandomState',
        completed_episodes=panel['episode_count'], teacher_action_probability=0, optimizer_steps=0,
        raytraced_fallback_allowed=False, corrected_initial_reset_required=True,
        augmentation_applied=False, offline_BC_teacher50_equivalent=False)
    if config['schema'] in rendered_POOL_CONFIG_SCHEMAS:
        value['renderer_endpoint_pool'] = rendered_deepcopy(config['renderer'])
    return {**value, 'sha256': rendered_digest(value)}


def rendered_validate_rendered_result(panel, result, checkpoint_sha256, config):
    config = rendered_validate_execution(panel, checkpoint_sha256, config, arm=result.get('arm'))
    if result.get('rendered_student_success_protocol') != rendered_protocol_identity(panel, config):
        raise ValueError('rendered student result lacks exact renderer/configuration protocol')
    rendered_validate_result(panel, result, checkpoint_sha256, **rendered_evaluation_options(panel))
    allowed_hashes = {a['sha256'] for a in config['anchors']['files']}
    for index, (spec, receipt) in enumerate(zip(panel['cohorts'], result['rendered_observation_receipts'])):
        # validate_result above checked every receipt's structural/count/native
        # evidence. Bind to once-validated full config without 148 inventory scans.
        rendered__validate_frozen_cohort_binding(receipt, panel, config, index, allowed_hashes)
    return rendered_deepcopy(result)


def rendered_configuration(reference, panel, checkpoint_sha256):
    from pipeline.bc_training.handoff_common import bound
    if not isinstance(reference, dict) or set(reference) != {'path', 'raw_sha256'}:
        raise ValueError('hash-bound rendered_observation_config dependency template required')
    return rendered_bind_observation_config(bound(reference), panel, checkpoint_sha256)


def rendered_artifact(reference, panel, checkpoint_sha256):
    config = rendered_configuration(reference, panel, checkpoint_sha256)
    raw = rendered_canonical(config) + b'\n'
    return config, raw, rendered_hashlib.sha256(raw).hexdigest()


def rendered_payload_fields(relative_path, raw_sha256):
    path = rendered_Path(relative_path)
    if (path.is_absolute() or '..' in path.parts or path.suffix != '.json'
            or len(raw_sha256) != 64 or any(c not in '0123456789abcdef' for c in raw_sha256)):
        raise ValueError('safe jobs-relative rendered configuration and exact digest required')
    return dict(rendered_observation_config_file=str(path),
                rendered_observation_config_sha256=raw_sha256)


# ---- scaled success contract ----
from copy import deepcopy as scaled_deepcopy
import hashlib as scaled_hashlib
import re as scaled_re

from pipeline.deployment.rtx_actor import explicit_motion_panel as scaled_explicit
from pipeline.success_rate import production_success_contract as scaled_original
from pipeline.deployment.rtx_actor.initial_robot_refresh_diagnostic import SCALED_STUDENT_CLEAR_ONCE as scaled_SCALED_STUDENT_CLEAR_ONCE, control_contract as scaled_control_contract

scaled_PREFIX = 'scaled-rendered295-'
scaled_SCHEMA = 'roborender.scaled-rendered-success-protocol.v1'
scaled_DISTRIBUTION_SCHEMA = 'roborender.scaled-rendered-initialization.v1'
scaled_ADMISSION_SCHEMA = 'roborender.scaled-rendered-success-admission.v1'
scaled_PARTITION_RESULT_SCHEMA = 'roborender.scaled-rendered-success-partition.v1'
scaled_INITIAL_SCHEMA = 'roborender.scaled-rendered-initial-state.v1'
scaled_OPERATION = 'evaluate_scaled_rendered_panel'
scaled_BATCH_SIZES = (4, 8, 16)
scaled_INITIAL_ARRAYS = {'robot_root', 'dof_pos', 'dof_vel', 'stage_buf', 'episode_length_buf',
    'time_in_stage_buf', 'sample_idx', 'sample_idx_in_seq', 'front_object', 'hold_object',
    'tray_object', 'actor_obs', 'teacher_obs', 'homie_obs', 'gt_obj_pos'}
scaled_CHECKPOINT_SHA256 = '2ec87a3c7b76aaa8fbb57d4fd50d02335dbfed75c164f5d741bea0067a33d7a9'


def scaled_digest(value):
    return scaled_hashlib.sha256(scaled_explicit.canonical(value)).hexdigest()


def scaled__sha(value):
    return isinstance(value, str) and scaled_re.fullmatch(r'[0-9a-f]{64}', value) is not None


def scaled_is_panel(value):
    return isinstance(value, dict) and isinstance(value.get('panel_id'), str) and value['panel_id'].startswith(scaled_PREFIX)


def scaled_raytraced_shape(panel):
    """Structural parser only, also used on receipt-checking singleton copies."""
    if not scaled_is_panel(panel) or panel.get('observation_mode') != 'image3f_3view':
        raise ValueError('explicit scaled rendered panel required')
    value = scaled_deepcopy(panel)
    value.pop('scaled_protocol', None)
    value['observation_mode'] = 'raytraced_3view'
    return value


def scaled__build(source, batch_size, qualification, qualification_cohorts, qualification_variant='paired'):
    if type(batch_size) is not int or batch_size not in scaled_BATCH_SIZES or type(qualification) is not bool:
        raise ValueError('scaled evaluation admits explicit B4/B8/B16 only')
    if (qualification_variant not in ('paired', 'tail7')
            or (qualification_variant == 'tail7' and (not qualification or batch_size != 16))):
        raise ValueError('tail7 qualification is an explicit B16-profile qualification variant only')
    expected_cohorts = 1 if qualification_variant == 'tail7' else 2
    if type(qualification_cohorts) is not int or qualification_cohorts != expected_cohorts:
        raise ValueError('qualification v1 fixes two full cohorts per batch')
    purpose = 'qualification' if qualification else 'test'
    identifier = f'{scaled_PREFIX}b{batch_size}-{purpose}{"-tail7" if qualification_variant == "tail7" else ""}-v1'
    cases = [(case, lane) for cohort in source['cohorts']
             for case, lane in zip(cohort['case_ids'], cohort['assignment']['lanes'])]
    selected = (cases[-7:] if qualification_variant == 'tail7' else
                cases[:2 * batch_size] if qualification else cases)
    value = {key: scaled_deepcopy(item) for key, item in source.items() if key != 'cohorts'}
    # Qualification uses a separate protocol/seed namespace; native task purpose
    # remains test because it never collects training or held-out MSE data.
    value.update(panel_id=identifier, episode_count=len(selected), cohorts=[])
    qualification_offset = 200000 if qualification_variant == 'tail7' else 100000 if qualification else 0
    seed_base = 880000 + batch_size * 1000 + qualification_offset
    for index, start in enumerate(range(0, len(selected), batch_size)):
        group = selected[start:start + batch_size]
        assignment = dict(schema='roborender.native-motion-cohort.v1', run_id=identifier,
            cohort_id=f'{identifier}-cohort-{index:03d}', purpose='test',
            simulator_seed=seed_base + index, motion_inventory_sha256=source['motion_inventory_sha256'],
            lanes=[dict(scaled_deepcopy(lane), lane=i) for i, (_, lane) in enumerate(group)])
        value['cohorts'].append(dict(assignment=assignment,
            case_ids=[f'{identifier}-case-{start + i:04d}' for i in range(len(group))]))
    value['scaled_protocol'] = dict(schema=scaled_DISTRIBUTION_SCHEMA, batch_size=batch_size,
        purpose=purpose, qualification_cohorts=qualification_cohorts,
        seed_base=seed_base, source_panel=scaled_deepcopy(source), source_panel_sha256=scaled_explicit.panel_sha256(source),
        source_case_ids=[case for case, _ in selected],
        historical_initial_states_preserved=False, historical_outcomes_reused=False,
        reset_distribution='original-motion-order-regrouped-native-reset-rng-v1',
        seed_rule='880000 + 1000*batch_size + (100000 if qualification else 0) + cohort_index',
        sample_indices_forced=False, scene_origins_depend_on_batch=True)
    if qualification_variant == 'tail7':
        value['scaled_protocol'].update(qualification_variant='tail7',
            seed_rule='880000 + 1000*batch_size + 200000 + cohort_index',
            source_case_selection='original-last-seven-motion-cases-only')
    return value


def scaled_build_panel(source_panel, *, batch_size=16, qualification=False, qualification_cohorts=2,
                qualification_variant='paired'):
    source = scaled_original.require_panel(source_panel)
    if qualification_variant == 'tail7' and qualification_cohorts == 2:
        qualification_cohorts = 1
    return scaled_require_panel(scaled__build(source, batch_size, qualification, qualification_cohorts, qualification_variant))


def scaled_require_panel(panel):
    if not scaled_is_panel(panel) or not isinstance(panel.get('scaled_protocol'), dict):
        raise ValueError('scaled evaluation requires its new versioned distribution')
    metadata = panel['scaled_protocol']
    source = scaled_original.require_panel(metadata.get('source_panel'))
    expected = scaled__build(source, metadata.get('batch_size'), metadata.get('purpose') == 'qualification',
                      metadata.get('qualification_cohorts'), metadata.get('qualification_variant', 'paired'))
    if scaled_explicit.canonical(panel) != scaled_explicit.canonical(expected):
        raise ValueError('scaled motion order, seeds, case identity or distribution changed')
    scaled_explicit.validate_panel(scaled_raytraced_shape(panel))
    return scaled_deepcopy(panel)


def scaled_evaluation_options(panel):
    scaled_require_panel(panel)
    return dict(teacher_action_mode='distill', diagnostic_initial_robot_refresh=scaled_SCALED_STUDENT_CLEAR_ONCE)


def scaled_protocol_identity(panel):
    panel = scaled_require_panel(panel)
    metadata = panel['scaled_protocol']
    value = dict(schema=scaled_SCHEMA, panel_sha256=scaled_explicit.panel_sha256(panel),
        initialization_distribution_sha256=scaled_digest(metadata), source_panel_sha256=metadata['source_panel_sha256'],
        batch_size=metadata['batch_size'], purpose=metadata['purpose'],
        expected_episodes=panel['episode_count'], cohort_sizes=[len(c['case_ids']) for c in panel['cohorts']],
        qualification_only=metadata['purpose'] == 'qualification', historical_fixed295_equivalent=False,
        historical_outcomes_reused=False, observation_mode='image3f_3view', raytraced_fallback_allowed=False,
        checkpoint_selection='immutable_admitted_student_checkpoint', behavior='deterministic_student_action_mean_11_plus_homie_15',
        teacher_action_probability=0, optimizer_steps=0, max_episode_seconds=8, control_hz=50, physics_hz=200,
        metric='native_goal_or_until_done', goal_accounting_version=scaled_explicit.GOAL_ACCOUNTING_VERSION,
        reset_control=scaled_control_contract(scaled_SCALED_STUDENT_CLEAR_ONCE),
        stop_boundary='one_complete_cohort_per_native_command')
    return dict(value, sha256=scaled_digest(value))


def scaled__configuration(panel, config, *, arm='bc'):
    from pipeline.success_rate.rendered_student_success_contract import validate_execution, POOL_CONFIG_SCHEMAS
    scaled_require_panel(panel)
    config = validate_execution(panel, config.get('checkpoint_sha256') if isinstance(config, dict) else None, config, arm=arm)
    if config['schema'] not in POOL_CONFIG_SCHEMAS:
        raise ValueError('scaled evaluator requires the pinned endpoint-pool configuration')
    return config


def scaled_single_partition(panel, cohort_indices):
    scaled_require_panel(panel)
    if not isinstance(cohort_indices, list) or len(cohort_indices) != 1:
        raise ValueError('scaled commands must stop at one complete-cohort boundary')
    return scaled_explicit.make_partition(panel, cohort_indices)


def scaled_build_admission(panel, config, *, source_sha256, worker_id, command_id, output_name,
                    cohort_indices, qualification_evidence_sha256=None, arm='bc'):
    config = scaled__configuration(panel, config, arm=arm)
    partition = scaled_single_partition(panel, cohort_indices)
    if (not scaled__sha(source_sha256) or not isinstance(worker_id, str) or not scaled_re.fullmatch(r'[0-9a-f]{32}', worker_id)
            or any(not isinstance(v, str) or not scaled_re.fullmatch(r'[a-z0-9][a-z0-9-]{0,95}', v)
                   for v in (command_id, output_name))):
        raise ValueError('scaled admission must bind exact source, owner, command and output')
    qualification = panel['scaled_protocol']['purpose'] == 'qualification'
    if ((qualification and qualification_evidence_sha256 is not None)
            or (not qualification and not scaled__sha(qualification_evidence_sha256))):
        raise ValueError('test launch requires actual qualification evidence; qualification is a separate admission')
    return dict(schema=scaled_ADMISSION_SCHEMA, operation=scaled_OPERATION, source_sha256=source_sha256,
        worker_instance_id=worker_id, command_id=command_id, output_name=output_name,
        panel_sha256=scaled_explicit.panel_sha256(panel), config_sha256=scaled_digest(config),
        partition_sha256=scaled_digest(partition), cohort_indices=list(cohort_indices),
        checkpoint_sha256=config['checkpoint_sha256'], arm=arm, protocol_sha256=scaled_protocol_identity(panel)['sha256'],
        qualification_evidence_sha256=qualification_evidence_sha256,
        qualification_only=qualification, retain_all_h200_compute=True,
        simulator_snapshot_restore=False, historical_outcomes_reused=False)


def scaled_validate_admission(admission, panel, config, *, source_sha256, native_state, output_name, partition, arm='bc'):
    if not isinstance(admission, dict) or not isinstance(native_state, dict):
        raise ValueError('scaled execution requires explicit admission and current native state')
    expected = scaled_build_admission(panel, config, source_sha256=source_sha256,
        worker_id=native_state.get('worker_instance_id'), command_id=admission.get('command_id'),
        output_name=output_name, cohort_indices=partition.get('cohort_indices'),
        qualification_evidence_sha256=admission.get('qualification_evidence_sha256'), arm=arm)
    if (admission != expected or native_state.get('source_sha256') != source_sha256
            or partition != scaled_single_partition(panel, admission['cohort_indices'])):
        raise ValueError('scaled admission differs from actual source, owner, panel, configuration or partition')
    phase, active = native_state.get('phase'), native_state.get('active_command_id')
    commands = native_state.get('commands')
    if not isinstance(commands, dict) or phase not in {'IDLE', 'RUNNING'}:
        raise ValueError('scaled native owner is busy or uncertain')
    cid = admission['command_id']
    if phase == 'IDLE' and (active is not None or cid in commands):
        raise ValueError('scaled command cannot replay an accepted identity')
    if phase == 'RUNNING':
        row = commands.get(cid, {})
        command = row.get('command', {})
        if (active != cid or row.get('status') != 'RUNNING' or command.get('operation') != scaled_OPERATION
                or command.get('worker_instance_id') != admission['worker_instance_id']
                or command.get('payload', {}).get('output_name') != output_name):
            raise ValueError('scaled command is not the exact active admitted operation')
    return scaled_deepcopy(admission)


def scaled_validate_initial_attestation(value, panel, index):
    spec = panel['cohorts'][index]
    if (not isinstance(value, dict) or set(value) != {'schema', 'native_reset', 'initial_state',
            'environment_step_calls_before_attestation', 'initialization_distribution_sha256'}
            or value['schema'] != scaled_INITIAL_SCHEMA or value['environment_step_calls_before_attestation'] != 0
            or value['initialization_distribution_sha256'] != scaled_digest(panel['scaled_protocol'])):
        raise ValueError('scaled cohort lacks its before-action initialization proof')
    reset = value['native_reset']
    if (not isinstance(reset, dict) or set(reset) != {'schema', 'assignment', 'assignment_sha256', 'lanes',
            'sample_indices_forced', 'seed_reapplied_per_episode'}
            or reset['schema'] != 'roborender.explicit-motion-success-reset.v1'
            or reset['assignment'] != spec['assignment'] or reset['assignment_sha256'] != scaled_digest(spec['assignment'])
            or reset['sample_indices_forced'] is not False or reset['seed_reapplied_per_episode'] is not False
            or not isinstance(reset['lanes'], list) or len(reset['lanes']) != len(spec['case_ids'])):
        raise ValueError('scaled native reset assignment differs')
    for lane, row in zip(spec['assignment']['lanes'], reset['lanes']):
        if (not isinstance(row, dict) or set(row) != set(lane) | {'sample_idx', 'sample_idx_in_seq', 'reset_rng_sha256'}
                or any(row[k] != v for k, v in lane.items())
                or any(type(row[k]) is not int or row[k] < 0 for k in ('sample_idx', 'sample_idx_in_seq'))
                or not scaled__sha(row['reset_rng_sha256'])):
            raise ValueError('scaled actual reset sample/RNG evidence differs')
    states = value['initial_state']
    if not isinstance(states, dict) or set(states) != scaled_INITIAL_ARRAYS:
        raise ValueError('scaled initial state digests missing')
    for descriptor in states.values():
        if (not isinstance(descriptor, dict) or set(descriptor) != {'sha256', 'shape', 'dtype'}
                or not scaled__sha(descriptor['sha256']) or not isinstance(descriptor['shape'], list)
                or not descriptor['shape'] or descriptor['shape'][0] != len(spec['case_ids'])
                or any(type(v) is not int or v < 0 for v in descriptor['shape'])
                or not isinstance(descriptor['dtype'], str)):
            raise ValueError('scaled initial state digest malformed')
    return scaled_deepcopy(value)


def scaled_validate_cohort_observations(result, **kwargs):
    # Explicit-motion structural validators use the same renderer receipts.
    from pipeline.success_rate.rendered_student_success_contract import validate_cohort_observations as check
    return check(result, **kwargs)


def scaled_wrap_partition(panel, partition_result, checkpoint_sha256, rendered_config):
    value = dict(schema=scaled_PARTITION_RESULT_SCHEMA, complete=True, full_panel_complete=False,
        scaled_success_protocol=scaled_protocol_identity(panel), configuration_sha256=scaled_digest(rendered_config),
        checkpoint_sha256=checkpoint_sha256, partition_result=scaled_deepcopy(partition_result),
        qualification_only=panel['scaled_protocol']['purpose'] == 'qualification')
    return scaled_validate_result(panel, value, checkpoint_sha256, rendered_config=rendered_config)


def scaled_validate_result(panel, result, checkpoint_sha256, *, rendered_config):
    arm = result.get('partition_result', {}).get('arm') if isinstance(result, dict) and result.get('schema') == scaled_PARTITION_RESULT_SCHEMA else result.get('arm') if isinstance(result, dict) else None
    config = scaled__configuration(panel, rendered_config, arm=arm)
    if checkpoint_sha256 != config['checkpoint_sha256'] or not isinstance(result, dict):
        raise ValueError('scaled result must evaluate the exact selected student')
    if result.get('scaled_success_protocol') != scaled_protocol_identity(panel):
        raise ValueError('scaled result protocol/distribution differs')
    options = scaled_evaluation_options(panel)
    if result.get('schema') == scaled_PARTITION_RESULT_SCHEMA:
        value = result.get('partition_result')
        if not isinstance(value, dict):
            raise ValueError('scaled result missing completed-cohort partition')
        partition = scaled_single_partition(panel, value.get('partition', {}).get('cohort_indices'))
        checked = scaled_explicit._partition_result(panel, partition, value.get('cohort_receipts'),
            checkpoint_sha256=checkpoint_sha256, arm=arm, **options)
        expected = dict(schema=scaled_PARTITION_RESULT_SCHEMA, complete=True, full_panel_complete=False,
            scaled_success_protocol=scaled_protocol_identity(panel), configuration_sha256=scaled_digest(config),
            checkpoint_sha256=checkpoint_sha256, partition_result=checked,
            qualification_only=panel['scaled_protocol']['purpose'] == 'qualification')
        if result != expected:
            raise ValueError('scaled partition result identity or accounting differs')
        for receipt in checked['cohort_receipts']:
            index, row = receipt['cohort_index'], receipt['result']
            scaled_validate_cohort_observations(row, count=len(panel['cohorts'][index]['case_ids']),
                config=config, cohort_index=index, panel=panel)
            scaled_validate_initial_attestation(row.get('scaled_initial_state_attestation'), panel, index)
    else:
        from pipeline.success_rate.rendered_student_success_contract import validate_rendered_result
        validate_rendered_result(panel, result, checkpoint_sha256, config)
        attestations = result.get('scaled_initial_state_attestations')
        if not isinstance(attestations, list) or len(attestations) != len(panel['cohorts']):
            raise ValueError('scaled aggregate initialization proof incomplete')
        for index, row in enumerate(attestations):
            scaled_validate_initial_attestation(row, panel, index)
        if result.get('qualification_only') != (panel['scaled_protocol']['purpose'] == 'qualification'):
            raise ValueError('qualification outcomes cannot be reported as the production test')
    return scaled_deepcopy(result)


def scaled_aggregate_results(panel, results, checkpoint_sha256, rendered_config, *, arm='bc'):
    from pipeline.success_rate.rendered_student_success_contract import protocol_identity as rendered_identity
    checked = [scaled_validate_result(panel, result, checkpoint_sha256, rendered_config=rendered_config)
               for result in results]
    if any(row['schema'] != scaled_PARTITION_RESULT_SCHEMA for row in checked):
        raise ValueError('only exact complete-cohort partitions may be merged')
    partitions = [row['partition_result'] for row in checked]
    result = scaled_explicit.aggregate_partitions(panel, partitions, checkpoint_sha256=checkpoint_sha256,
        arm=arm, **scaled_evaluation_options(panel))
    by_index = {receipt['cohort_index']: receipt['result']['scaled_initial_state_attestation']
                for part in partitions for receipt in part['cohort_receipts']}
    result.update(scaled_success_protocol=scaled_protocol_identity(panel),
        rendered_student_success_protocol=rendered_identity(panel, rendered_config),
        scaled_initial_state_attestations=[by_index[i] for i in range(len(panel['cohorts']))],
        qualification_only=panel['scaled_protocol']['purpose'] == 'qualification')
    return scaled_validate_result(panel, result, checkpoint_sha256, rendered_config=rendered_config)


# ---- scaled teacher benchmark ----
from copy import deepcopy as teacher_deepcopy
import hashlib as teacher_hashlib
import re as teacher_re

try:
    from pipeline.success_rate import scaled_success_contract as teacher_scaled
    from pipeline.deployment.rtx_actor import explicit_motion_panel as teacher_explicit
    from pipeline.deployment.rtx_actor import initial_robot_refresh_diagnostic as teacher_reset
    from pipeline.deployment.rtx_actor.teacher_action_diagnostic import RAVEN_PPO as teacher_RAVEN_PPO, control_contract as teacher_control_contract, validate_parameter_attestation as teacher_validate_parameter_attestation
except ImportError:
    import scaled_success_contract as teacher_scaled
    import explicit_motion_panel as teacher_explicit
    import initial_robot_refresh_diagnostic as teacher_reset
    from teacher_action_diagnostic import RAVEN_PPO as teacher_RAVEN_PPO, control_contract as teacher_control_contract, validate_parameter_attestation as teacher_validate_parameter_attestation

teacher_SCHEMA = 'roborender.scaled-teacher-benchmark.v1'
teacher_RESULT_SCHEMA = 'roborender.scaled-teacher-cohort-result.v1'
teacher_AGGREGATE_SCHEMA = 'roborender.scaled-teacher-full295-result.v1'
teacher_OPERATION = 'evaluate_scaled_teacher_panel'
teacher_OWNER_FIELDS = ('physical_gpu', 'gpu_uuid', 'native_pid', 'worker_instance_id', 'source_sha256')


def teacher_digest(value):
    return teacher_hashlib.sha256(teacher_explicit.canonical(value)).hexdigest()


def teacher_require_panel(panel):
    panel = teacher_scaled.require_panel(panel)
    if (panel['scaled_protocol']['purpose'] != 'test' or panel['scaled_protocol']['batch_size'] != 16
            or panel['episode_count'] != 295
            or [len(c['case_ids']) for c in panel['cohorts']] != [16]*18+[7]):
        raise ValueError('teacher baseline requires the complete unchanged scaled B16 fixed295 panel')
    return panel


def teacher_evaluation_options():
    return dict(arm='teacher', teacher_action_mode=teacher_RAVEN_PPO,
                diagnostic_initial_robot_refresh=teacher_reset.SCALED_TEACHER_CLEAR_ONCE)


def teacher_build_benchmark(panel, *, owner, command_id, output_name, cohort_index, shadow_checkpoint_sha256):
    panel = teacher_require_panel(panel)
    if type(cohort_index) is not int or not 0 <= cohort_index < 19:
        raise ValueError('teacher command must name exactly one original cohort')
    if not isinstance(owner, dict) or any(k not in owner for k in teacher_OWNER_FIELDS):
        raise ValueError('exact native owner identity required')
    selected = {k: owner[k] for k in teacher_OWNER_FIELDS}
    if (type(selected['physical_gpu']) is not int or selected['physical_gpu'] != cohort_index % 2
            or type(selected['native_pid']) is not int or selected['native_pid'] < 1
            or not isinstance(selected['gpu_uuid'], str) or not selected['gpu_uuid'].startswith('GPU-')
            or teacher_re.fullmatch('[a-f0-9]{32}', str(selected['worker_instance_id'])) is None
            or teacher_re.fullmatch('[a-f0-9]{64}', str(selected['source_sha256'])) is None
            or teacher_re.fullmatch('[a-f0-9]{64}', str(shadow_checkpoint_sha256)) is None
            or any(not isinstance(v, str) or teacher_re.fullmatch('[a-z0-9][a-z0-9-]{0,95}', v) is None
                   for v in (command_id, output_name))):
        raise ValueError('teacher command source/owner/checkpoint identity differs')
    spec = panel['cohorts'][cohort_index]
    return dict(schema=teacher_SCHEMA, operation=teacher_OPERATION, owner=selected, command_id=command_id,
        output_name=output_name, **teacher_evaluation_options(), cohort_index=cohort_index,
        panel_sha256=teacher_explicit.panel_sha256(panel), physical_cohort_sha256=teacher_digest(spec),
        initialization_distribution_sha256=teacher_digest(panel['scaled_protocol']),
        shadow_not_behavior_checkpoint_sha256=shadow_checkpoint_sha256,
        teacher_control=teacher_control_contract(teacher_RAVEN_PPO), num_envs=len(spec['case_ids']),
        full_panel_expected_episodes=295, observation_mode='privileged_state',
        camera_observations_consumed_by_behavior=False, renderer_calls=0, student_forward_calls=0,
        optimizer_steps=0, initial_clocks='fixed295_fresh_clocks', training_curriculum_restored=False,
        metric='native_goal_or_until_first_done', max_vector_steps=450,
        historical_B2_initial_states_preserved=False, historical_outcomes_reused=False,
        reset_control=teacher_reset.control_contract(teacher_reset.SCALED_TEACHER_CLEAR_ONCE),
        goal_accounting_version=teacher_explicit.GOAL_ACCOUNTING_VERSION,
        native_replay_allowed=False, compute_release_authorized=False)


def teacher_validate_benchmark(value, panel):
    if not isinstance(value, dict):
        raise ValueError('separately admitted teacher benchmark required')
    expected = teacher_build_benchmark(panel, owner=value.get('owner'), command_id=value.get('command_id'),
        output_name=value.get('output_name'), cohort_index=value.get('cohort_index'),
        shadow_checkpoint_sha256=value.get('shadow_not_behavior_checkpoint_sha256'))
    if value != expected:
        raise ValueError('teacher benchmark science/identity changed')
    return teacher_deepcopy(value)


def teacher_validate_owner(value, state, *, source_sha256, physical_gpu, native_pid):
    owner = value['owner']
    if (owner['source_sha256'] != source_sha256 or owner['physical_gpu'] != physical_gpu
            or owner['native_pid'] != native_pid
            or any(state.get(k) != owner[k] for k in teacher_OWNER_FIELDS if k != 'physical_gpu')):
        raise ValueError('teacher native lifetime/source differs')
    phase, active, commands = state.get('phase'), state.get('active_command_id'), state.get('commands')
    cid = value['command_id']
    if not isinstance(commands, dict) or phase not in {'IDLE', 'RUNNING'}:
        raise ValueError('teacher owner is uncertain')
    if phase == 'IDLE' and (active is not None or cid in commands):
        raise ValueError('teacher benchmark cannot replay accepted work')
    if phase == 'RUNNING':
        row = commands.get(cid, {}); command = row.get('command', {})
        if (active != cid or row.get('status') != 'RUNNING' or command.get('operation') != teacher_OPERATION
                or command.get('worker_instance_id') != owner['worker_instance_id']
                or command.get('payload', {}).get('output_name') != value['output_name']):
            raise ValueError('teacher benchmark is not the exact active operation')


def teacher_wrap_result(panel, benchmark, result):
    teacher_validate_benchmark(benchmark, panel)
    value = dict(schema=teacher_RESULT_SCHEMA, complete=True, full_panel_complete=False,
        benchmark=teacher_deepcopy(benchmark), benchmark_sha256=teacher_digest(benchmark), observation_mode='privileged_state',
        cohort_result=teacher_deepcopy(result))
    return teacher_validate_result(panel, value)


def teacher_validate_result(panel, value):
    panel = teacher_require_panel(panel)
    if (not isinstance(value, dict) or set(value) != {'schema','complete','full_panel_complete','benchmark',
            'benchmark_sha256','observation_mode','cohort_result'} or value['schema'] != teacher_RESULT_SCHEMA
            or value['complete'] is not True or value['full_panel_complete'] is not False
            or value['observation_mode'] != 'privileged_state'):
        raise ValueError('separate teacher cohort result required')
    benchmark = teacher_validate_benchmark(value['benchmark'], panel)
    if value['benchmark_sha256'] != teacher_digest(benchmark):
        raise ValueError('teacher result descriptor digest differs')
    row = value['cohort_result']; index = benchmark['cohort_index']
    # Existing raw native receipt labels available cameras, not policy input.
    # Reuse its exact first-episode/goal/parameter/reset checks, never a rendered validator.
    teacher_explicit.make_cohort_receipt(teacher_scaled.raytraced_shape(panel), index, row,
        checkpoint_sha256=benchmark['shadow_not_behavior_checkpoint_sha256'], **teacher_evaluation_options())
    teacher_scaled.validate_initial_attestation(row.get('scaled_initial_state_attestation'), panel, index)
    for key in ('teacher_parameter_attestation', 'teacher_parameter_attestation_at_start'):
        teacher_validate_parameter_attestation(row.get(key))
    try:
        from pipeline.deployment.rtx_actor.isolated_ppo_teacher import validate_execution_receipt
    except ImportError:
        from isolated_ppo_teacher import validate_execution_receipt
    validate_execution_receipt(row.get('teacher_isolation'), vector_steps=row.get('vector_steps'))
    isolation = row.get('teacher_isolation') or {}
    if (isolation.get('shared_teacher_unchanged') is not True or isolation.get('private_teacher_used') is not True
            or isolation.get('student_forward_calls') != 0 or isolation.get('renderer_calls') != 0
            or row.get('teacher_control') != benchmark['teacher_control']
            or type(row.get('vector_steps')) is not int or not 1 <= row['vector_steps'] <= 450
            or row.get('original_raven_metric_completed') != benchmark['num_envs']
            or row.get('original_raven_metric_successes') != row.get('exported_successes')
            or row.get('training_curriculum_restored') is not False
            or type(row.get('teacher_resource_guard_checks')) is not int
            or row.get('teacher_resource_guard_checks', 0) < 2*row['vector_steps']
            or any(type(r.get('timeout')) is not bool or r['steps'] > row['vector_steps'] for r in row['outcomes'])
            or 'rendered_observation_receipt' in row):
        raise ValueError('teacher-only execution/isolation/guard evidence differs')
    return teacher_deepcopy(value)


def teacher_aggregate_results(panel, results):
    panel = teacher_require_panel(panel)
    if not isinstance(results, list) or len(results) != 19:
        raise ValueError('full295 teacher result requires all19 completed cohorts')
    values = [teacher_validate_result(panel, v) for v in results]
    if [v['benchmark']['cohort_index'] for v in values] != list(range(19)):
        raise ValueError('teacher cohort order, coverage or duplication differs')
    source = {v['benchmark']['owner']['source_sha256'] for v in values}
    shadows = {v['benchmark']['shadow_not_behavior_checkpoint_sha256'] for v in values}
    owners = {gpu: {teacher_digest(v['benchmark']['owner']) for v in values if v['benchmark']['owner']['physical_gpu']==gpu}
              for gpu in (0,1)}
    if (len(source)!=1 or len(shadows)!=1 or any(len(v)!=1 for v in owners.values())
            or len({v['benchmark']['command_id'] for v in values}) != 19
            or len({v['benchmark']['output_name'] for v in values}) != 19
            or len({v['benchmark']['owner']['gpu_uuid'] for v in values}) != 2
            or len({v['benchmark']['owner']['worker_instance_id'] for v in values}) != 2):
        raise ValueError('teacher aggregate changed source, shadow checkpoint or native owner')
    summary = teacher_explicit.summarize(teacher_scaled.raytraced_shape(panel), [v['cohort_result'] for v in values])
    return dict(schema=teacher_AGGREGATE_SCHEMA, complete=True, arm='teacher', observation_mode='privileged_state',
        panel_sha256=teacher_explicit.panel_sha256(panel), source_sha256=next(iter(source)),
        teacher_control=teacher_control_contract(teacher_RAVEN_PPO), shadow_not_behavior_checkpoint_sha256=next(iter(shadows)),
        completed=295, successes=summary['successes'], success_rate=summary['success_rate'],
        outcomes=summary['outcomes'], cohort_results=teacher_deepcopy(values), optimizer_steps=0,
        renderer_calls=0, student_forward_calls=0, historical_outcomes_reused=False,
        compute_release_authorized=False)


# ---- episode protocol ----
from copy import deepcopy as pooled_deepcopy
import hashlib as pooled_hashlib
import json as pooled_json
import math as pooled_math
import re as pooled_re

pooled_SCHEMA = 'roborender.raven-pooled-rendered-benchmark.v1'
pooled_RESULT_SCHEMA = 'roborender.raven-pooled-rendered-result.v1'
pooled_OPERATION = 'evaluate_raven_pooled_rendered'
pooled_CLOCK_SCHEMA = 'roborender.raven-random-initial-clocks.v1'
pooled_OWNER_FIELDS = ('physical_gpu', 'gpu_uuid', 'native_pid', 'worker_instance_id', 'source_sha256')
pooled_SOURCE_PINS = {
    'trl/trainer/distill_trainer_obj_pred_homie_api.py': '3da5a2557ae18a8ba53c4da14d8841b80a609b851b326a21534ee2c5001b7584',
    'envs/loco_manip/walk_stand_grasp.py': 'd985d3e74414b6ceabefd917bdaa3b36c67ff9e141882436f931815f16bd3be9',
    'envs/loco_manip/reset_from_dataset_wsppt.py': 'd42185ef7535e193bb5affcda6e7ad2fbac92e1c478feb9a846ed455f476ac94',
    'utils/teleop_reset_lib/dataset_wsppt.py': '949cc147ad9abc32600e277fb69a0579ad5fd87e7f2086249785b966d88de1a3',
}


def pooled_canonical(value):
    return pooled_json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def pooled_digest(value):
    return pooled_hashlib.sha256(pooled_canonical(value)).hexdigest()


def pooled_build_benchmark(panel, config, *, owner, command_id, output_name, cohort_index,
                    qualification_only, arm='dagger'):
    from pipeline.success_rate import scaled_success_contract as scaled
    from pipeline.deployment.rtx_actor.rendered_endpoint_pool import expected_physical_gpu
    panel = scaled.require_panel(panel)
    if arm not in {'bc','dagger'}:
        raise ValueError('pooled benchmark admits frozen student arms only')
    scaled._configuration(panel, config, arm=arm)
    if (type(qualification_only) is not bool or type(cohort_index) is not int
            or cohort_index not in (0, 1) or len(panel['cohorts']) != 2
            or panel['scaled_protocol']['batch_size'] != 16
            or panel['scaled_protocol']['purpose'] != 'qualification'
            or any(len(c['assignment']['lanes']) != 16 for c in panel['cohorts'])):
        raise ValueError('pooled benchmark requires two declared B16 motion-support replicates')
    selected = {key: owner[key] for key in pooled_OWNER_FIELDS}
    if (selected['physical_gpu'] != cohort_index or type(selected['physical_gpu']) is not int
            or type(selected['native_pid']) is not int or selected['native_pid'] <= 0
            or not isinstance(selected['gpu_uuid'], str) or not selected['gpu_uuid'].startswith('GPU-')
            or pooled_re.fullmatch('[a-f0-9]{32}', selected['worker_instance_id']) is None
            or pooled_re.fullmatch('[a-f0-9]{64}', selected['source_sha256']) is None
            or expected_physical_gpu(config['renderer'], cohort_index) != cohort_index
            or any(not isinstance(v, str) or pooled_re.fullmatch('[a-z0-9][a-z0-9-]{0,95}', v) is None
                   for v in (command_id, output_name))):
        raise ValueError('pooled benchmark requires exact disjoint native/renderer owner identities')
    target = 32 if qualification_only else 150
    return dict(schema=pooled_SCHEMA, operation=pooled_OPERATION, owner=selected, command_id=command_id, arm=arm,
        output_name=output_name, qualification_only=qualification_only, num_envs=16,
        minimum_completed_episodes=target, max_vector_steps=450 * pooled_math.ceil(target / 16),
        cohort_index=cohort_index, panel_sha256=scaled.explicit.panel_sha256(panel),
        config_sha256=pooled_digest(config), checkpoint_sha256=config['checkpoint_sha256'],
        motion_assignment_sha256=pooled_digest(panel['cohorts'][cohort_index]['assignment']),
        motion_support=pooled_deepcopy(panel['cohorts'][cohort_index]['assignment']['lanes']),
        motion_sampling='fixed_motion_per_lane_native_reference_frame_resampling',
        full295_motion_coverage=False, motion_support_size_per_actor=16,
        population='independent_per_actor_B16_pooled_completed_episodes',
        stopping='at_next_loop_check_after_all_terminal_lanes_counted',
        initial_clocks=pooled_CLOCK_SCHEMA, observations_recomputed_after_clock_randomization=False,
        natural_episode_clocks_randomized=False, training_curriculum_restored=False,
        evaluation_curriculum='cold_existing_corrected_evaluation_configuration',
        generated_only=True, teacher_probability=0, optimizer_steps=0,
        fixed295_equivalent=False, published_numeric_equivalence=False,
        corrected_initial_reset_retained=True, terminal_goal_fix_retained=True,
        source_pins=pooled_deepcopy(pooled_SOURCE_PINS), compute_release_authorized=False,
        native_replay_allowed=False)


def pooled_validate_benchmark(value, panel, config):
    if not isinstance(value, dict):
        raise ValueError('explicit separately versioned pooled benchmark required')
    expected = pooled_build_benchmark(panel, config, owner=value.get('owner', {}),
        command_id=value.get('command_id'), output_name=value.get('output_name'),
        cohort_index=value.get('cohort_index'), qualification_only=value.get('qualification_only'), arm=value.get('arm'))
    if value != expected:
        raise ValueError('pooled benchmark science/identity changed')
    return pooled_deepcopy(value)


def pooled_validate_owner(value, state, *, source_sha256, physical_gpu, native_pid):
    owner = value['owner']
    if (owner['source_sha256'] != source_sha256 or owner['physical_gpu'] != physical_gpu
            or owner['native_pid'] != native_pid
            or any(state.get(k) != owner[k] for k in pooled_OWNER_FIELDS if k != 'physical_gpu')):
        raise ValueError('pooled benchmark native lifetime/source differs')
    phase, active, commands = state.get('phase'), state.get('active_command_id'), state.get('commands')
    cid = value['command_id']
    if not isinstance(commands, dict) or phase not in {'IDLE', 'RUNNING'}:
        raise ValueError('pooled benchmark owner is uncertain')
    if phase == 'IDLE' and (active is not None or cid in commands):
        raise ValueError('pooled benchmark cannot replay accepted work')
    if phase == 'RUNNING':
        row = commands.get(cid, {}); command = row.get('command', {})
        if (active != cid or row.get('status') != 'RUNNING' or command.get('operation') != pooled_OPERATION
                or command.get('worker_instance_id') != owner['worker_instance_id']
                or command.get('payload', {}).get('output_name') != value['output_name']):
            raise ValueError('pooled benchmark is not the exact active operation')


def pooled_randomize_initial_clocks(env):
    """Pinned Raven order, no reset, no observations recomputed, no extra draws."""
    import torch
    count = env.num_envs
    names = ('episode_length_buf', 'time_in_stage_buf', 'actual_time_in_stage_buf')
    if (any(not isinstance(getattr(env, key, None), torch.Tensor)
                   or list(getattr(env, key).shape) != [count] for key in names)):
        raise ValueError('one staged native initial-clock initialization required')
    before = {name: getattr(env, name).detach().cpu().tolist() for name in names}
    env.episode_length_buf = torch.randint_like(env.episode_length_buf, high=int(env.max_episode_length))
    env.time_in_stage_buf[:] = torch.randint_like(env.time_in_stage_buf, high=int(env.max_stage_time[0]))
    env.actual_time_in_stage_buf[:] = env.time_in_stage_buf
    return dict(schema=pooled_CLOCK_SCHEMA, num_envs=count, order=list(names), before=before,
        after={name: getattr(env, name).detach().cpu().tolist() for name in names},
        episode_high=int(env.max_episode_length), stage_high=int(env.max_stage_time[0]),
        dtype={name: str(getattr(env, name).dtype) for name in names},
        after_corrected_reset=True, before_metrics_initialization=True,
        observations_recomputed=False, physics_steps_added=0)


from .episode_metrics import PooledEpisodeAccounting as pooled_PooledEpisodeAccounting


def pooled_validate_result(result, benchmark):
    if (not isinstance(result, dict) or result.get('schema') != pooled_RESULT_SCHEMA
            or result.get('benchmark') != benchmark or result.get('benchmark_sha256') != pooled_digest(benchmark)
            or result.get('complete') is not True or result.get('checkpoint_sha256') != benchmark['checkpoint_sha256']
            or result.get('optimizer_steps') != 0 or result.get('policy_unchanged') is not True
            or result.get('observation_mode') != 'image3f_3view' or result.get('benchmark_certified') is not False
            or result.get('training_curriculum_restored') is not False
            or result.get('teacher_action_mode') != 'distill' or result.get('arm') != benchmark['arm']
            or result.get('accounting_disagreements') != 0):
        raise ValueError('separate pooled native benchmark result required')
    rows = result.get('outcomes'); count = result.get('completed'); target = benchmark['minimum_completed_episodes']
    if (type(count) is not int or not target <= count <= target + 15 or not isinstance(rows, list)
            or len(rows) != count or result.get('overshoot') != count-target
            or result.get('expected') != target or type(result.get('vector_steps')) is not int
            or not 1 <= result['vector_steps'] <= benchmark['max_vector_steps']):
        raise ValueError('pooled final-step denominator/overshoot differs')
    indices = [0] * 16; previous_terminal = [-1] * 16; last = (-1, -1); successes = 0
    for index, row in enumerate(rows):
        lane, step = row.get('lane'), row.get('terminal_vector_step')
        if (type(lane) is not int or not 0 <= lane < 16 or type(step) is not int
                or not 0 <= step < result['vector_steps'] or (step, lane) <= last
                or row.get('completion_index') != index or row.get('episode_index') != indices[lane]
                or type(row.get('steps')) is not int or row['steps'] != step-previous_terminal[lane]
                or type(row.get('timeout')) is not bool
                or type(row.get('exported_success')) is not bool or type(row.get('native_success')) is not bool):
            raise ValueError('pooled ordered episode identities differ')
        indices[lane] += 1; previous_terminal[lane] = step; last = (step, lane); successes += row['exported_success']
    final_step = rows[-1]['terminal_vector_step']
    if (sum(row['terminal_vector_step'] < final_step for row in rows) >= target
            or final_step != result['vector_steps'] - 1 or result.get('episodes_per_lane') != indices
            or result.get('exported_successes') != successes
            or result.get('native_successes') != successes
            or any(row['exported_success'] != row['native_success'] for row in rows)
            or result.get('original_raven_metric_completed') != count
            or result.get('original_raven_metric_successes') != successes):
        raise ValueError('pooled stop boundary or original metric differs')
    from pipeline.deployment.rtx_actor import initial_robot_refresh_diagnostic as reset
    from pipeline.deployment.rtx_actor.explicit_motion_panel import GOAL_ACCOUNTING_VERSION
    if (result.get(reset.FIELD) != reset.SCALED_STUDENT_CLEAR_ONCE
            or result.get('initial_robot_refresh_control') != reset.control_contract(reset.SCALED_STUDENT_CLEAR_ONCE)
            or result.get('goal_accounting_version') != GOAL_ACCOUNTING_VERSION):
        raise ValueError('pooled benchmark must retain corrected initial reset and terminal goal semantics')
    reset.validate_attestation(result.get('initial_robot_refresh_attestation'), num_envs=16, mode=reset.SCALED_STUDENT_CLEAR_ONCE)
    clock = result.get('initial_clocks', {})
    if (clock.get('schema') != pooled_CLOCK_SCHEMA or clock.get('num_envs') != 16
            or clock.get('after_corrected_reset') is not True or clock.get('before_metrics_initialization') is not True
            or clock.get('observations_recomputed') is not False or clock.get('physics_steps_added') != 0):
        raise ValueError('actual randomized initial-clock observation required')
    for key, high in [('episode_length_buf', clock.get('episode_high')), ('time_in_stage_buf', clock.get('stage_high'))]:
        values = clock.get('after', {}).get(key)
        if type(high) is not int or high <= 0 or not isinstance(values, list) or len(values) != 16 or any(
                type(v) not in (int, float) or not pooled_math.isfinite(v) or int(v) != v or not 0 <= v < high for v in values):
            raise ValueError('randomized initial clocks are outside original half-open bounds')
    if clock['after'].get('actual_time_in_stage_buf') != clock['after']['time_in_stage_buf']:
        raise ValueError('actual stage clock must equal randomized stage clock')
    rendered = result.get('rendered_observation_receipt', {})
    if (rendered.get('vector_steps_rendered') != result['vector_steps']
            or rendered.get('render_requests_completed') != 16*result['vector_steps']
            or rendered.get('raytraced_fallbacks') != 0 or rendered.get('failed') is not False):
        raise ValueError('pooled benchmark requires every lane rendered through final step')
    expected_anchors = [(0, lane, 0) for lane in range(16)] + [
        (row['terminal_vector_step']+1, row['lane'], row['episode_index']+1)
        for row in rows if row['terminal_vector_step'] < final_step]
    expected_anchors.sort()
    anchors = rendered.get('anchor_selections', [])
    if [(r.get('selected_at_vector_step'), r.get('lane'), r.get('episode_index')) for r in anchors] != expected_anchors:
        raise ValueError('pooled anchors must reset exactly on every natural episode boundary before the next action')
    namespace = pooled_digest(benchmark)
    if any(r.get('episode_session_id') != f"{namespace}:lane{r['lane']}:episode{r['episode_index']}" for r in anchors):
        raise ValueError('pooled renderer episode namespaces differ')
    sessions = rendered.get('pooled_episode_sessions', {})
    current = [max(r['episode_index'] for r in anchors if r['lane'] == lane) for lane in range(16)]
    if sessions != dict(benchmark_sha256=namespace, current_episode_indices=current,
            natural_done_refreshes_anchor=True, generated_frame_history_carried=False,
            transport_request_sequence='monotonic_not_scientific_episode_state'):
        raise ValueError('pooled render-session lifecycle differs')
    resets = result.get('native_episode_resets')
    if not isinstance(resets, list) or len(resets) != count+16:
        raise ValueError('pooled result requires initial and every natural native reset')
    reset_counts = [0]*16
    for row in resets:
        lane = row.get('lane')
        if (type(lane) is not int or not 0 <= lane < 16 or row.get('episode_index') != reset_counts[lane]
                or row.get('motion_id') != benchmark['motion_support'][lane]['motion_id']
                or any(type(row.get(k)) is not int or row[k] < 0 for k in ('sample_idx','sample_idx_in_seq'))
                or pooled_re.fullmatch('[a-f0-9]{64}', row.get('reset_rng_sha256', '')) is None):
            raise ValueError('pooled actual native reset identity differs')
        reset_counts[lane] += 1
    if reset_counts != [n+1 for n in indices] or type(result.get('native_reset_resample_every')) is not int or result['native_reset_resample_every'] <= 0:
        raise ValueError('pooled native reset/resampling coverage differs')
    return pooled_deepcopy(result)
