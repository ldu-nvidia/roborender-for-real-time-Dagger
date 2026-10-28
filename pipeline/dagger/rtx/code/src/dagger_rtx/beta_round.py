"""Versioned beta production contract shared by the CPU and native actor.

This is deliberately separate from both alpha production and the immutable
three-window qualification. Loading a config does not allocate or launch GPUs.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from .contracts import ROUND_BETAS, require_sha256

SCHEMA = "roborender.cross_gpu_dagger_round.beta.v1"
SMOKE_SCHEMA = "roborender.cross_gpu_dagger_smoke.beta.v1"
LEO_SCHEMA = "roborender.cross_gpu_dagger_round.leo.v1"
LEO_SMOKE_SCHEMA = "roborender.cross_gpu_dagger_smoke.leo.v1"
LEO_MSE_SMOKE_SCHEMA = "roborender.cross_gpu_dagger_mse_smoke.leo.v1"
LEO_MSE_SMOKE_CORRECTED_SCHEMA = "roborender.cross_gpu_dagger_mse_smoke.leo.v2"
LEO_MSE_PRODUCTION_SCHEMA = "roborender.cross_gpu_dagger_mse_full295.leo.v1"
LEO_MSE_SCALED_SCHEMA = "roborender.cross_gpu_dagger_mse_full295.leo.v2"
LEO_FIXED100_SCHEMA = "roborender.cross_gpu_dagger_fixed100_full295.leo.v1"
LEO_CONTINUOUS_FIXED100_SCHEMA = "roborender.cross_gpu_dagger_continuous_fixed100.leo.v1"
LEO_CONTINUOUS_FIXED100_8H200_SCHEMA = "roborender.cross_gpu_dagger_continuous_fixed100.leo.v2"
SCHEMAS = {SCHEMA, SMOKE_SCHEMA, LEO_SCHEMA, LEO_SMOKE_SCHEMA, LEO_MSE_SMOKE_SCHEMA, LEO_MSE_SMOKE_CORRECTED_SCHEMA, LEO_MSE_PRODUCTION_SCHEMA, LEO_MSE_SCALED_SCHEMA, LEO_FIXED100_SCHEMA, LEO_CONTINUOUS_FIXED100_SCHEMA}
SCHEMAS.add(LEO_CONTINUOUS_FIXED100_8H200_SCHEMA)
MODE = "beta-production"
ACK = "I_ACKNOWLEDGE_BETA_FIFTY_WINDOWS_AND_REAL_LEARNER_UPDATES"
SMOKE_ACK = "I_ACKNOWLEDGE_BETA_THREE_SMOKE_WINDOWS_AND_REAL_LEARNER_UPDATES"
GPU_UUIDS = {1: "GPU-0c1949ae-379a-2f2e-0f23-621de2eee409",
             2: "GPU-0540f144-9826-ba8f-140d-af4f95a475ce"}
LEO_GPU_UUIDS = {0: "GPU-4d58bc2b-43c8-6ff8-e7fd-2348575f0cff",
                 1: "GPU-b92877a7-a5e0-9a83-3bc0-d8ede481f35d"}
FIDELITY = {"raytraced_native_augmentation": True, "full_batch_sync_batchnorm": True,
            "random_initial_episode_lengths": True}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def is_leo(config):
    return config.get("schema") in {LEO_SCHEMA, LEO_SMOKE_SCHEMA, LEO_MSE_SMOKE_SCHEMA, LEO_MSE_SMOKE_CORRECTED_SCHEMA, LEO_MSE_PRODUCTION_SCHEMA, LEO_MSE_SCALED_SCHEMA, LEO_FIXED100_SCHEMA, LEO_CONTINUOUS_FIXED100_SCHEMA, LEO_CONTINUOUS_FIXED100_8H200_SCHEMA}


def is_eight_h200(config):
    return isinstance(config, dict) and config.get('schema') == LEO_CONTINUOUS_FIXED100_8H200_SCHEMA


def gateway_slots(config, physical_gpu):
    if physical_gpu not in gpu_uuids(config):
        raise ValueError('gateway route needs an admitted physical RTX GPU')
    return list(range(physical_gpu*4, physical_gpu*4+4)) if is_eight_h200(config) else list(range(4))


def gateway_pids_for_actor(config, physical_gpu):
    return [config['h200']['gateway_pids'][i] for i in gateway_slots(config, physical_gpu)]


def gateway_identities_for_actor(config, physical_gpu):
    return [config['h200']['gateway_identities'][i] for i in gateway_slots(config, physical_gpu)] if is_eight_h200(config) else []


def verify_actor_gateways(config, physical_gpu, ready):
    """Join live PID/server identity to PREPARE's attested GPU/source rows."""
    expected = gateway_identities_for_actor(config, physical_gpu)
    if not expected:
        return []
    observed = ready.get('gateways', [])
    if len(observed) != 4 or any(
            actual.get('pid') != row['pid'] or actual.get('gateway_instance_id') != row['gateway_instance_id']
            for actual, row in zip(observed, expected)):
        raise ValueError('live exclusive H200 gateway PID/server identities differ from PREPARE')
    return [dict(row) for row in expected]


def is_continuous_fixed100(config):
    return isinstance(config, dict) and config.get("schema") in {LEO_CONTINUOUS_FIXED100_SCHEMA, LEO_CONTINUOUS_FIXED100_8H200_SCHEMA}


def is_fixed100(config):
    return isinstance(config, dict) and config.get("schema") in {LEO_FIXED100_SCHEMA, LEO_CONTINUOUS_FIXED100_SCHEMA, LEO_CONTINUOUS_FIXED100_8H200_SCHEMA}


def is_smoke(config):
    return config.get("schema") in {SMOKE_SCHEMA, LEO_SMOKE_SCHEMA, LEO_MSE_SMOKE_SCHEMA, LEO_MSE_SMOKE_CORRECTED_SCHEMA}


def is_mse_smoke(config):
    return isinstance(config, dict) and config.get("schema") in {LEO_MSE_SMOKE_SCHEMA, LEO_MSE_SMOKE_CORRECTED_SCHEMA}


def is_corrected_smoke(config):
    return isinstance(config, dict) and config.get("schema") == LEO_MSE_SMOKE_CORRECTED_SCHEMA


def is_mse_production(config):
    # Includes the new fixed-budget protocol only for production-layout routing;
    # its separate schema must never activate MSE stopping or held-out collection.
    return isinstance(config, dict) and config.get("schema") in {LEO_MSE_PRODUCTION_SCHEMA, LEO_MSE_SCALED_SCHEMA, LEO_FIXED100_SCHEMA, LEO_CONTINUOUS_FIXED100_SCHEMA, LEO_CONTINUOUS_FIXED100_8H200_SCHEMA}


def uses_explicit_motion(config):
    return is_mse_smoke(config) or is_mse_production(config)


def environments_per_actor(config):
    if is_fixed100(config):
        return 16
    if config.get('schema')==LEO_MSE_SCALED_SCHEMA:
        actors=config.get('rtx_actors')
        if not isinstance(actors,list) or len(actors)!=2:
            raise ValueError('two scaled actors required')
        batch=actors[0].get('concurrent_environments')
        if type(batch)is not int or batch not in (2,4,8,16,32,64) or actors[1].get('concurrent_environments')!=batch:
            raise ValueError('same bounded environment batch required on both RTX GPUs')
        return batch
    return 2 if uses_explicit_motion(config) else 6


def gpu_uuids(config):
    if config.get("schema") not in SCHEMAS:
        raise ValueError("unknown host-bound round schema")
    return dict(LEO_GPU_UUIDS if is_leo(config) else GPU_UUIDS)


def deployment_host(config):
    return "leo-rtx-6k" if is_leo(config) else "465d6b0-lcedt"


def host_role(config):
    return "leo" if is_leo(config) else "beta"


def budgets(config):
    """Smoke is a distinct immutable contract, never a partial production run."""
    if is_fixed100(config):
        return 100, 295, config.get("validation", {}).get("panel_protocol")
    if is_mse_production(config):
        validation = config.get("validation", {})
        if config.get('schema')==LEO_MSE_SCALED_SCHEMA:
            # V2 remains ten complete coverage epochs; no fixed1480 update
            # budget survives a change of full-environment batch size.
            total=2*environments_per_actor(config)
            return 20*((295+total-1)//total),295,validation.get('panel_protocol')
        return 1480, 295, validation.get("panel_protocol")
    if is_mse_smoke(config):
        validation = config.get("validation", {})
        return 3, validation.get("completed_episodes"), validation.get("panel_protocol")
    if is_smoke(config):
        return 3, 10, "beta-isaac6-b5-heldout-10-smoke-v1"
    if config.get("schema") in {SCHEMA, LEO_SCHEMA}:
        return 50, 256, "beta-isaac6-b5-heldout-256-v4"
    raise ValueError("unknown beta contract")


def acknowledgement(config):
    if is_continuous_fixed100(config):
        return "I_ACKNOWLEDGE_ONE_CONTINUOUS_NATIVE_ROUND_AND_100_FULL_UPDATES"
    if is_mse_production(config):
        return "I_ACKNOWLEDGE_LEO_FULL295_COHORT_LIFECYCLES_AND_MSE_SELECTION"
    if is_leo(config):
        return "I_ACKNOWLEDGE_LEO_" + ("THREE_SMOKE" if is_smoke(config) else "FIFTY") + "_WINDOWS_AND_REAL_LEARNER_UPDATES"
    return SMOKE_ACK if is_smoke(config) else ACK


def validate_config(config, *, runtime=False):
    required = {"schema", "run_id", "round_id", "teacher_beta", "student_action_probability", "seed",
                "source_sha256", "rtx_actors", "forbidden_gpu_indices", "raven_fidelity", "replay",
                "execution", "h200", "student", "validation", "logging"}
    if is_mse_smoke(config):
        required.add("motion_assignments")
    if is_mse_production(config):
        required.add("coverage_plan")
    if is_fixed100(config):
        required.add("student_observation")
    if not isinstance(config, dict) or not required <= set(config) or set(config) - required - {"existing_run_config_sha256", "learner_protocol", "behavior_selection_protocol", "population_curriculum_protocol", "reset_inventory_protocol"}:
        raise ValueError("unknown or missing beta round fields")
    if 'reset_inventory_protocol' in config and (
            not is_continuous_fixed100(config)
            or config['reset_inventory_protocol'] != 'roborender.raven-full-reset-inventory.v1'):
        raise ValueError('full reset inventory requires a new continuous campaign')
    if 'learner_protocol' in config and (
            not is_continuous_fixed100(config)
            or config['learner_protocol'] != 'roborender.raven-environment-randperm.v1'):
        raise ValueError('Raven learner ordering requires a new continuous campaign')
    if 'behavior_selection_protocol' in config and (
            not is_continuous_fixed100(config)
            or config['behavior_selection_protocol'] != 'roborender.raven-native-vector-selection.v1'):
        raise ValueError('native vector selection requires a new continuous campaign')
    if 'population_curriculum_protocol' in config and (
            not is_continuous_fixed100(config)
            or config['population_curriculum_protocol'] != 'roborender.raven-paired-population-curriculum.v1'):
        raise ValueError('shared curriculum requires a new continuous campaign')
    if config.get("schema") not in SCHEMAS:
        raise ValueError("not a beta production round")
    if is_fixed100(config):
        from .observation_contract import validate_generated_only
        validate_generated_only(config["student_observation"])
    windows, episodes, panel_protocol = budgets(config)
    role = host_role(config)
    if is_smoke(config) and not config.get("run_id", "").startswith(f"dagger-{role}-smoke-"):
        raise ValueError("smoke runs require a visibly separate smoke identity")
    if not re.fullmatch(r"dagger-" + role + r"-[a-z0-9-]{1,64}", config.get("run_id", "")):
        raise ValueError("a separate beta production run identity is required")
    rid = config.get("round_id")
    if type(rid) is not int or rid not in ROUND_BETAS or config.get("teacher_beta") != ROUND_BETAS[rid]:
        raise ValueError("round/beta must follow teacher75/50/25/0")
    if config.get("student_action_probability") != 1 - ROUND_BETAS[rid]:
        raise ValueError("student probability must complement teacher beta")
    if type(config.get("seed")) is not int or not 0 <= config["seed"] < 2**31 - 2:
        raise ValueError("invalid simulator seed")
    actors = config.get("rtx_actors")
    expected = [{"physical_gpu_index": gpu, "gpu_uuid": uuid, "concurrent_environments": environments_per_actor(config)}
                for gpu, uuid in gpu_uuids(config).items()]
    forbidden = [2] if is_leo(config) else [0]
    if canonical(actors) != canonical(expected) or canonical(config.get("forbidden_gpu_indices")) != canonical(forbidden):
        raise ValueError("host-bound GPU identities, versioned environment batch, and exclusions must match")
    if is_mse_smoke(config):
        assignments = config["motion_assignments"]
        if (not isinstance(assignments, list) or len(assignments) != 2
                or [row.get("physical_gpu_index") for row in assignments] != [0, 1]):
            raise ValueError("MSE smoke requires two explicit native training assignments")
        for row in assignments:
            if set(row) != {"physical_gpu_index", "path", "sha256"} or not Path(row["path"]).is_absolute():
                raise ValueError("training assignments must be absolute hash-bound artifacts")
            require_sha256(row["sha256"], "native training assignment")
    if is_mse_production(config):
        ref = config["coverage_plan"]
        if (not isinstance(ref, dict) or set(ref) != {"path", "raw_sha256"}
                or not Path(ref["path"]).is_absolute()):
            raise ValueError("production requires an explicit frozen full295 coverage plan")
        require_sha256(ref["raw_sha256"], "production coverage plan")
    if canonical(config.get("raven_fidelity")) != canonical(FIDELITY) or canonical(config.get("replay")) != b'{"enabled":false}':
        raise ValueError("beta requires native augmentation, full batch, random phases and no replay")
    if canonical(config.get("execution")) != canonical({
        "window_count": windows, "rollout_steps": 81, "wait_for_checkpoint_advance": True,
        "learner_submission_authorized": True, "validation_authorized": True,
    }):
        raise ValueError("beta round budget differs from its versioned contract")
    h200 = config.get("h200", {})
    eight = is_eight_h200(config)
    count = 8 if eight else 4
    h200_fields = {"render_student_worker_count", "boundary_contract", "resource_shape", "reservation", "allow_burst", "gateway_pids"}
    if eight:
        h200_fields |= {'actor_gateway_slots', 'gateway_identities'}
    if set(h200) != h200_fields:
        raise ValueError("unknown or missing H200 configuration")
    if (type(h200.get("render_student_worker_count")) is not int or h200.get("render_student_worker_count") != count
            or h200.get("boundary_contract") != "roborender.dagger.rtx.v3"
            or h200.get("resource_shape") != f"custom.gpu.{count}xh200"
            or h200.get("reservation") != "cosmos-solution-architect-xcwtffiq"
            or h200.get("allow_burst") is not False):
        raise ValueError("H200 fleet count/shape must match its explicit schema")
    pids = h200.get("gateway_pids", [])
    if pids == [] and not runtime:
        pass  # Unbound planning template; PREPARE must attest actual PIDs.
    elif len(pids) != count or len(set(pids)) != count or any(type(p) is not int or p < 1 for p in pids):
        raise ValueError("all distinct live gateway PIDs must be pinned")
    if eight:
        if h200['actor_gateway_slots'] != {'0': [0,1,2,3], '1': [4,5,6,7]}:
            raise ValueError('eight-H200 requires four exclusive renderer slots per RTX')
        identities = h200['gateway_identities']
        if identities == [] and pids == [] and not runtime:
            pass
        else:
            if not isinstance(identities, list) or len(identities) != 8:
                raise ValueError('all eight actual gateway identities must be pinned')
            for slot, row in enumerate(identities):
                if (set(row) != {'slot','pid','gateway_instance_id','gpu_uuid','source_sha256'}
                        or type(row['slot']) is not int or row['slot'] != slot
                        or type(row['pid']) is not int or row['pid'] != pids[slot] or row['source_sha256'] != config['source_sha256']
                        or not isinstance(row['gpu_uuid'], str) or not row['gpu_uuid'].startswith('GPU-')
                        or not isinstance(row['gateway_instance_id'], str)
                        or not re.fullmatch('[0-9a-f]{64}', row['gateway_instance_id'])):
                    raise ValueError('eight-H200 gateway slot/PID/server/source binding differs')
            if any(len({r[k] for r in identities}) != 8 for k in ('gpu_uuid','gateway_instance_id')):
                raise ValueError('eight distinct H200 GPUs and server instances required')
    student = config.get("student", {})
    if set(student) != {"initial_checkpoint", "initial_checkpoint_sha256", "learning_rate", "max_gradient_norm",
                        "mini_batches_per_iteration", "activation_checkpointing"}:
        raise ValueError("unknown or missing student configuration")
    if (student.get("learning_rate") != .0002 or student.get("max_gradient_norm") != 1.
            or student.get("mini_batches_per_iteration") != 1
            or student.get("activation_checkpointing") is not False):
        raise ValueError("learner must use one full versioned environment batch with the qualified settings")
    if not Path(student.get("initial_checkpoint", "")).is_absolute():
        raise ValueError("initial checkpoint must be an explicit host-local path")
    validation = config.get("validation", {})
    rendered_test = uses_explicit_motion(config)
    validation_fields = {"interval_iterations", "completed_episodes", "metric", "observation_mode", "panel_protocol",
                         "scenario_manifest", "scenario_manifest_sha256"}
    if rendered_test:
        validation_fields.add('rendered_observation_config')
    if set(validation) != validation_fields:
        raise ValueError("unknown or missing evaluation configuration")
    if rendered_test:
        ref = validation['rendered_observation_config']
        if (not isinstance(ref, dict) or set(ref) != {'path', 'raw_sha256'}
                or not Path(ref.get('path', '')).is_absolute() or '..' in Path(ref['path']).parts):
            raise ValueError('rendered final test requires an immutable dependency template binding')
        require_sha256(ref['raw_sha256'], 'rendered success dependencies')
    if is_mse_smoke(config) and (type(episodes) is not int or not 1 <= episodes <= 64
            or not isinstance(panel_protocol, str) or not panel_protocol.startswith("explicit-")):
        raise ValueError("explicit MSE smoke requires a separately named bounded motion panel")
    if is_mse_production(config) and (not isinstance(panel_protocol, str) or not panel_protocol):
        raise ValueError("production requires the separately frozen full295 native test")
    if (validation.get("interval_iterations") != windows or validation.get("completed_episodes") != episodes
            or validation.get("metric") != "raven_goal_reached_rate"
            or validation.get("observation_mode") != ('generated' if rendered_test else 'raytraced')
            or validation.get("panel_protocol") != panel_protocol):
        raise ValueError("beta final evaluation differs from its versioned panel/budget")
    if not Path(validation.get("scenario_manifest", "")).is_absolute():
        raise ValueError("panel path must be explicit")
    for value, name in ((student.get("initial_checkpoint_sha256"), "checkpoint"),
                        (validation.get("scenario_manifest_sha256"), "panel"),
                        (config.get("source_sha256"), "source")):
        require_sha256(value, name)
    previous = config.get("existing_run_config_sha256")
    if rid == 1:
        if previous is not None:
            raise ValueError("round1 must register a fresh coordinator identity")
    else:
        require_sha256(previous, "original coordinator config")
    logging = config.get("logging", {})
    if set(logging) != {"wandb_project", "wandb_run_name", "wandb_run_id"}:
        raise ValueError("unknown or missing W&B configuration")
    expected_suffix = f"-r{rid}-teacher{int(100 * ROUND_BETAS[rid])}"
    expected_name = (f"2026-10-16-prod-1-dagger-teacher{int(100 * ROUND_BETAS[rid])}"
                     if is_mse_production(config) else config["run_id"] + expected_suffix)
    if config.get('schema') in {LEO_MSE_SCALED_SCHEMA, LEO_FIXED100_SCHEMA, LEO_CONTINUOUS_FIXED100_SCHEMA, LEO_CONTINUOUS_FIXED100_8H200_SCHEMA}:
        expected_name=f"{config['run_id'].removeprefix('dagger-leo-')}-dagger-teacher{int(100*ROUND_BETAS[rid])}"
    if (logging.get("wandb_project") != "dagger"
            or logging.get("wandb_run_name") != expected_name
            or not re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", logging.get("wandb_run_id", ""))):
        raise ValueError("distinct round/teacher-labelled W&B identity required")
    return config


def actor_ids(config):
    return [f"{config['run_id']}-g{row['physical_gpu_index']}-lane-{lane:03d}"
            for row in config["rtx_actors"] for lane in range(row["concurrent_environments"])]


def read_config(path, expected_sha):
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha:
        raise ValueError("beta round configuration digest changed")
    return validate_config(json.loads(raw), runtime=True)


def driver_config(config, config_sha):
    """Normalized input to the existing persistent-window state machine."""
    validate_config(config, runtime=True)
    return {"run_id": config["run_id"], "round_id": config["round_id"], "windows": budgets(config)[0],
            "num_envs_per_gpu": environments_per_actor(config), "physical_rtx_gpus": list(gpu_uuids(config)),
            "initial_checkpoint_sha256": config["student"]["initial_checkpoint_sha256"],
            "teacher_beta": config["teacher_beta"],
            "coordinator_config_sha256": config.get("existing_run_config_sha256") or config_sha}
