"""Native explicit-motion success executor; imported only after Kit startup.

The caller owns scene lifecycle and must construct each scene inside the shared
explicit_scene_motion_binding context. This module never changes task physics,
termination equations, reset indices, or the student checkpoint.
"""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
import hashlib
import os
from pathlib import Path

import torch
from omegaconf import OmegaConf

from explicit_motion_panel import canonical, run_panel, validate_panel
from native_motion_assignment import assert_motion_environment, validate_motion_assignment
from native_artifact_io import publish_json
from isaac6_success_diagnostic import (
    IsaacSim as NormalizedEvaluationIsaacSim, NativeIsaacSim,
    SuccessDiagnosticTrainer,
)
from teacher_action_diagnostic import DISTILL, RAVEN_PPO, validate_mode, validate_trace_steps
from initial_robot_refresh_diagnostic import (
    ORIGINAL as ORIGINAL_INITIAL_REFRESH, STUDENT_CLEAR_ONCE, SMOKE_STUDENT_CLEAR_ONCE, validate_mode as validate_initial_refresh,
    validate_panel_mode as validate_initial_refresh_panel,
    require_corrected_execution,
)


class IsaacSim(NormalizedEvaluationIsaacSim):
    """Reuse exact RGB normalization, but bind scene metadata only via shared hook."""
    def __init__(self, config, *args, **kwargs):
        if config.reset_from_dataset.use_motion_file_dir:
            raise RuntimeError("explicit evaluation cannot use implicit motion filesystem order")
        # Do not run the legacy diagnostic constructor's global metadata hook.
        # The enclosing shared assignment context verifies bytes and lane order.
        NativeIsaacSim.__init__(self, config, *args, **kwargs)


@contextmanager
def _environment(values):
    saved = {key: os.environ.get(key) for key in values}
    os.environ.update({key: str(value) for key, value in values.items()})
    try:
        yield
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


class ExplicitMotionSuccessEvaluator(SuccessDiagnosticTrainer):
    def _diagnostic_expected_envs(self):
        return len(self.motion_assignment["lanes"])

    def _diagnostic_after_reset(self, obs, output):
        assignment = assert_motion_environment(self.env.config, self.motion_assignment)
        expected = hashlib.sha256(canonical(assignment)).hexdigest()
        if (getattr(self.env.simulator, "native_motion_assignment_sha256", None) != expected
                or list(self.env.reset_dataset.motion_files) != [r["path"] for r in assignment["lanes"]]
                or torch.initial_seed() != assignment["simulator_seed"]
                or self.env.config.domain_rand.image_augmentation.enabled):
            raise RuntimeError("native success scene/reset/seed/augmentation binding differs")
        ranges = self.env.reset_dataset.sample_idx_based_on_demo_clip_number
        if len(ranges) != len(assignment["lanes"]) or len(self._first_native_resets) != len(ranges):
            raise RuntimeError("success reset lacks one real sample per declared motion lane")
        rows = []
        for lane, motion in enumerate(assignment["lanes"]):
            row = self._first_native_resets[lane]
            if not int(ranges[lane][0]) <= row["sample_idx"] < int(ranges[lane][1]):
                raise RuntimeError("success reset sampled another lane's motion")
            rows.append({**motion, **row})
        reset_receipt = {
            "schema": "roborender.explicit-motion-success-reset.v1", "assignment": assignment,
            "assignment_sha256": expected, "lanes": rows,
            "sample_indices_forced": False, "seed_reapplied_per_episode": False}
        publish_json(output / "native-motion-reset.json", reset_receipt)
        if getattr(self, '_scaled_initialization_distribution', None) is not None:
            from isaac6_success_diagnostic import initial_state, digest
            from scaled_success_contract import INITIAL_SCHEMA
            self._scaled_initial_state_attestation = dict(schema=INITIAL_SCHEMA,
                native_reset=reset_receipt,
                initial_state={name: digest(value) for name, value in sorted(initial_state(self.env, obs).items())},
                environment_step_calls_before_attestation=0,
                initialization_distribution_sha256=self._scaled_initialization_distribution)
            publish_json(output / 'scaled-initial-state.json', self._scaled_initial_state_attestation)
        if getattr(self, "_continuation_initial_evidence", None) is not None:
            from isaac6_success_diagnostic import initial_state, digest, runtime_physics_evidence
            from rendered_success_continuation import compare_initial_boundary
            # SuccessDiagnosticTrainer strictly loads/checks every checkpoint
            # tensor before reset and invokes this hook before policy inference.
            # Compare the whole B2 state, including the unscored companion.
            self._continuation_initial_state_gate = compare_initial_boundary(
                self._continuation_initial_evidence, native_reset=reset_receipt,
                initial_state={name: digest(value) for name, value in sorted(initial_state(self.env, obs).items())},
                runtime_physics=runtime_physics_evidence(self.env),
                initial_refresh_attestation=getattr(self.env, "_diagnostic_initial_refresh_attestation", None),
                checkpoint_sha256=os.environ["DIAGNOSTIC_CHECKPOINT_SHA256"])
            publish_json(output / "continuation-initial-state-gate.json", self._continuation_initial_state_gate)
        if getattr(self, "_rendered_paired_baseline", None) is not None:
            from isaac6_success_diagnostic import initial_state, digest, runtime_physics_evidence
            from rendered_training_support_contract import compare_initial_state
            identity = {name: digest(value) for name, value in
                        sorted(initial_state(self.env, obs).items())}
            self._paired_initial_state_attestation = compare_initial_state(
                self._rendered_paired_baseline, cohort_index=self._rendered_cohort_index,
                initial_state=identity, runtime_physics=runtime_physics_evidence(self.env))
            publish_json(output / "paired-initial-state.json", self._paired_initial_state_attestation)

    def evaluate_cohort(self, assignment, *, checkpoint_path, checkpoint_sha256, output,
                        arm="bc", reference="", teacher_action_mode=DISTILL,
                        numeric_trace_steps=None, diagnostic_initial_robot_refresh=None):
        from rendered_success_observations import require_observation_transform
        require_observation_transform(self, arm=arm)
        validate_mode(teacher_action_mode, arm)
        require_corrected_execution(diagnostic_initial_robot_refresh, arm=arm,
            teacher_action_mode=teacher_action_mode, num_envs=len(assignment["lanes"]))
        from initial_robot_refresh_diagnostic import SCALED_STUDENT_CLEAR_ONCE
        if diagnostic_initial_robot_refresh in (STUDENT_CLEAR_ONCE, SMOKE_STUDENT_CLEAR_ONCE, SCALED_STUDENT_CLEAR_ONCE):
            from production_success_contract import require_native_timing
            require_native_timing(self.env)
        numeric_trace_steps = validate_trace_steps(
            (24 if teacher_action_mode == RAVEN_PPO else 0)
            if numeric_trace_steps is None else numeric_trace_steps)
        self.motion_assignment = validate_motion_assignment(assignment)
        if self.motion_assignment["purpose"] not in {"validation", "test"}:
            raise ValueError("success evaluation requires an explicit non-training purpose")
        output = Path(output)
        output.mkdir(parents=True, exist_ok=False)
        self.config = deepcopy(self.config)
        OmegaConf.set_struct(self.config, False)
        self.config.init_at_random_ep_len = False
        if getattr(self, '_raven_pooled_benchmark', None) is not None:
            self.config.init_at_random_ep_len = True
        self.config.raven_exact_raytraced_augmentation = False
        self.vg_raytraced_mix_prob = 1.0
        self._id_use_raytraced = torch.ones(len(assignment["lanes"]), dtype=torch.bool,
                                          device=self.accelerator.device)
        self._first_native_resets = {}
        if getattr(self, '_raven_pooled_benchmark', None) is not None:
            self._pooled_native_resets = []
            self._pooled_reset_counts = [0] * len(assignment['lanes'])
            self._pooled_latest_task_reset = [None] * len(assignment['lanes'])
        original = self.env._reset_task_states
        original_episode_reset = self.env.reset_envs_idx if getattr(self, '_raven_pooled_benchmark', None) is not None else None

        def observed_reset(lanes, *args, **kwargs):
            if len(lanes):
                fingerprint = hashlib.sha256(torch.get_rng_state().cpu().numpy().tobytes())
                fingerprint.update(torch.cuda.get_rng_state(self.env.device).cpu().numpy().tobytes())
            result = original(lanes, *args, **kwargs)
            if len(lanes):
                ids = lanes.detach().cpu().tolist()
                samples = self.env.sample_idx.detach().cpu().tolist()
                indices = self.env.sample_idx_in_seq.detach().cpu().tolist()
                if len(ids) != len(samples) or len(ids) != len(indices):
                    raise RuntimeError("native success reset sample cardinality differs")
                for lane, sample, index in zip(ids, samples, indices):
                    self._first_native_resets.setdefault(int(lane), {"sample_idx": int(sample),
                        "sample_idx_in_seq": int(index), "reset_rng_sha256": fingerprint.hexdigest()})
                    if getattr(self, '_raven_pooled_benchmark', None) is not None:
                        ranges = self.env.reset_dataset.sample_idx_based_on_demo_clip_number
                        if not int(ranges[lane][0]) <= int(sample) < int(ranges[lane][1]):
                            raise RuntimeError('pooled natural reset escaped declared lane motion')
                        self._pooled_latest_task_reset[lane] = dict(lane=int(lane), sample_idx=int(sample),
                            sample_idx_in_seq=int(index), reset_rng_sha256=fingerprint.hexdigest(),
                            motion_id=assignment['lanes'][lane]['motion_id'])
            return result

        def observed_episode_reset(lanes, *args, **kwargs):
            result = original_episode_reset(lanes, *args, **kwargs)
            for lane in lanes.detach().cpu().tolist():
                latest = self._pooled_latest_task_reset[lane]
                if latest is None:
                    raise RuntimeError('native episode reset lacks actual task sampling')
                self._pooled_native_resets.append(dict(latest, episode_index=self._pooled_reset_counts[lane]))
                if self._pooled_reset_counts[lane] == 0:
                    self._first_native_resets[lane] = {key: latest[key] for key in
                        ('sample_idx', 'sample_idx_in_seq', 'reset_rng_sha256')}
                self._pooled_reset_counts[lane] += 1
            return result

        self.env._reset_task_states = observed_reset
        if original_episode_reset is not None:
            self.env.reset_envs_idx = observed_episode_reset
        try:
            with _environment({"DIAGNOSTIC_ARM": arm, "DIAGNOSTIC_CHECKPOINT_PATH": checkpoint_path,
                    "DIAGNOSTIC_TEACHER_ACTION_MODE": teacher_action_mode,
                    "DIAGNOSTIC_NUMERIC_TRACE_STEPS": numeric_trace_steps,
                    "DIAGNOSTIC_INITIAL_ROBOT_REFRESH": diagnostic_initial_robot_refresh,
                    "DIAGNOSTIC_CHECKPOINT_SHA256": checkpoint_sha256,
                    "DIAGNOSTIC_OUTPUT_DIR": str(output), "DIAGNOSTIC_SEED": assignment["simulator_seed"],
                    "DIAGNOSTIC_RESEED_DATASET": "1", "DIAGNOSTIC_REFERENCE_DIR": reference}):
                result = super().train()
        finally:
            self.env._reset_task_states = original
            if original_episode_reset is not None:
                self.env.reset_envs_idx = original_episode_reset
        result["motion_assignment_sha256"] = hashlib.sha256(canonical(assignment)).hexdigest()
        if getattr(self, '_scaled_initialization_distribution', None) is not None:
            result['scaled_initial_state_attestation'] = deepcopy(self._scaled_initial_state_attestation)
        if getattr(self, "_continuation_initial_evidence", None) is not None:
            result["continuation_initial_state_gate"] = deepcopy(self._continuation_initial_state_gate)
        if getattr(self, "_rendered_paired_baseline", None) is not None:
            result["paired_initial_state_attestation"] = deepcopy(self._paired_initial_state_attestation)
        # The native diagnostic owns result.json and has already written it.
        # Preserve that original receipt; our augmented, assignment-bound receipt
        # is a separate immutable artifact (publish_json never overwrites).
        publish_json(output / "explicit-result.json", result)
        return result


def evaluate_explicit_panel(trainer, panel, *, root, checkpoint_path, checkpoint_sha256,
                            prepare_cohort, report, arm="bc", reference_root="",
                            completed_receipts=(), teacher_action_mode=DISTILL,
                            numeric_trace_steps=None, diagnostic_initial_robot_refresh=None,
                            training_support_selection=None, rendered_observation_config=None,
                            rendered_paired_baseline=None, _continuation=None, _scaled_partition=None,
                            _raven_benchmark=None, _teacher_benchmark=None):
    """Run cohorts in one warm owner. prepare_cohort must bind/reset fresh scenes.

    prepare_cohort(assignment) updates trainer.env and preserves the Kit owner.
    It must use the shared scene binder, a cold phase state, disabled image
    augmentation, and PanelResetEnvironment's qualified native reset semantics.
    """
    panel = validate_panel(panel)
    from scaled_success_contract import is_panel as is_scaled_panel
    from initial_robot_refresh_diagnostic import SCALED_STUDENT_CLEAR_ONCE, SCALED_TEACHER_CLEAR_ONCE
    if diagnostic_initial_robot_refresh == SCALED_TEACHER_CLEAR_ONCE and _teacher_benchmark is None:
        raise ValueError('scaled teacher reset requires its separate admitted teacher operation')
    if is_scaled_panel(panel) != (_scaled_partition is not None or _teacher_benchmark is not None):
        raise ValueError('scaled evaluation requires its separate admitted partition operation')
    if _teacher_benchmark is not None:
        from scaled_teacher_benchmark import validate_benchmark, evaluation_options as teacher_options
        validate_benchmark(_teacher_benchmark, panel)
        if (dict(arm=arm, teacher_action_mode=teacher_action_mode,
                 diagnostic_initial_robot_refresh=diagnostic_initial_robot_refresh) != teacher_options()
                or _scaled_partition is not None or _raven_benchmark is not None or _continuation is not None
                or completed_receipts or rendered_observation_config is not None or rendered_paired_baseline is not None
                or training_support_selection is not None or reference_root or numeric_trace_steps is not None
                or checkpoint_sha256 != _teacher_benchmark['shadow_not_behavior_checkpoint_sha256']
                or getattr(trainer, '_scaled_teacher_private', None) is None):
            raise ValueError('scaled teacher requires its separate fresh privileged-only operation')
    if _raven_benchmark is not None:
        from raven_pooled_benchmark import validate_benchmark
        validate_benchmark(_raven_benchmark, panel, rendered_observation_config)
        if (_scaled_partition is None or arm not in {'bc', 'dagger'} or _continuation is not None
                or _scaled_partition['cohort_indices'] != [_raven_benchmark['cohort_index']]):
            raise ValueError('pooled benchmark requires its distinct single-actor native command')
    if _scaled_partition is not None:
        from scaled_success_contract import require_panel as require_scaled_panel, single_partition
        require_scaled_panel(panel)
        if (_scaled_partition != single_partition(panel, _scaled_partition.get('cohort_indices'))
                or completed_receipts or _continuation is not None or training_support_selection is not None
                or rendered_paired_baseline is not None or rendered_observation_config is None
                or reference_root or numeric_trace_steps is not None):
            raise ValueError('scaled evaluation requires a fresh single-cohort generated-only operation')
    if _continuation is not None:
        from rendered_success_continuation import TRANSPORT
        evidence, admission = _continuation
        if (completed_receipts or arm != "bc" or training_support_selection is not None
                or rendered_paired_baseline is not None or rendered_observation_config is None
                or reference_root or numeric_trace_steps is not None
                or teacher_action_mode != DISTILL or diagnostic_initial_robot_refresh != STUDENT_CLEAR_ONCE
                or admission.get("evidence_sha256") != evidence.get("evidence_sha256")
                or admission.get("transport_recovery") != TRANSPORT):
            raise ValueError("continuation requires separate admitted fresh rendered BC execution")
    validate_mode(teacher_action_mode, arm)
    require_corrected_execution(diagnostic_initial_robot_refresh, arm=arm,
        teacher_action_mode=teacher_action_mode, has_resume=bool(completed_receipts))
    validate_initial_refresh_panel(panel, diagnostic_initial_robot_refresh, arm=arm,
        teacher_action_mode=teacher_action_mode, has_resume=bool(completed_receipts))
    rendered = rendered_observation_config is not None or rendered_paired_baseline is not None
    paired_rendered = rendered and (training_support_selection is not None or rendered_paired_baseline is not None)
    if paired_rendered:
        from rendered_training_support_contract import validate_execution, evaluation_options
        if (rendered_observation_config is None or rendered_paired_baseline is None
                or training_support_selection is None or reference_root or numeric_trace_steps is not None):
            raise ValueError("complete separate rendered/paired diagnostic inputs required")
        validate_execution(panel, training_support_selection, checkpoint_sha256,
            rendered_observation_config, rendered_paired_baseline, arm=arm, has_resume=bool(completed_receipts))
        required = evaluation_options(panel)
        if (teacher_action_mode != required["teacher_action_mode"]
                or diagnostic_initial_robot_refresh != required["diagnostic_initial_robot_refresh"]):
            raise ValueError("rendered diagnostic requires the unchanged corrected student protocol")
    elif rendered:
        from rendered_student_success_contract import validate_execution
        if reference_root or numeric_trace_steps is not None:
            raise ValueError("rendered student success requires a fresh independent execution")
        validate_execution(panel, checkpoint_sha256, rendered_observation_config,
                           arm=arm, has_resume=bool(completed_receipts))
        if _scaled_partition is not None:
            from rendered_endpoint_pool import expected_physical_gpu
            expected_gpu = expected_physical_gpu(rendered_observation_config['renderer'],
                                                 _scaled_partition['cohort_indices'][0])
            if expected_gpu is not None and os.environ.get('RESIDENT_PHYSICAL_GPU') != str(expected_gpu):
                raise ValueError('scaled evaluator actual physical GPU differs before scene preparation')
        if diagnostic_initial_robot_refresh == STUDENT_CLEAR_ONCE:
            from production_success_contract import evaluation_options
        elif diagnostic_initial_robot_refresh == SMOKE_STUDENT_CLEAR_ONCE:
            from smoke_success_contract import evaluation_options
        elif diagnostic_initial_robot_refresh == SCALED_STUDENT_CLEAR_ONCE:
            from scaled_success_contract import evaluation_options
        else:
            raise ValueError("rendered student success requires the corrected production/smoke protocol")
        required = evaluation_options(panel)
        if (teacher_action_mode != required["teacher_action_mode"]
                or diagnostic_initial_robot_refresh != required["diagnostic_initial_robot_refresh"]):
            raise ValueError("rendered student success options differ")
    elif arm in {"bc", "dagger"}:
        raise ValueError("student success requires rendered observation configuration before scene setup")
    elif training_support_selection is not None:
        from training_support_success_contract import validate_execution, evaluation_options
        validate_execution(panel, training_support_selection, checkpoint_sha256,
                           arm=arm, has_resume=bool(completed_receipts))
        required = evaluation_options(panel)
        if (teacher_action_mode != required["teacher_action_mode"]
                or diagnostic_initial_robot_refresh != required["diagnostic_initial_robot_refresh"]
                or reference_root or numeric_trace_steps is not None):
            raise ValueError("training-support evaluator requires its exact fresh student-only protocol")
    elif diagnostic_initial_robot_refresh == STUDENT_CLEAR_ONCE:
        from production_success_contract import evaluation_options
        evaluation_options(panel)
    elif diagnostic_initial_robot_refresh == SMOKE_STUDENT_CLEAR_ONCE:
        from smoke_success_contract import evaluation_options
        evaluation_options(panel)
    root = Path(root)
    if not root.resolve().is_relative_to("/dagger-runtime"):
        raise ValueError("success output must remain in the owned runtime")
    root.mkdir(parents=True, exist_ok=True)
    if any(root.iterdir()):
        raise ValueError("explicit success panel output must be fresh")
    receipts_root = root / "cohort-receipts"
    receipts_root.mkdir()
    # Resume is accounting at a certified whole-cohort boundary, never a
    # simulator snapshot restore. Pinning and validation happen again inside
    # run_panel before any new scene or policy step can execute.
    for receipt in completed_receipts:
        publish_json(receipts_root / f"cohort-{receipt['cohort_index']:03d}.json", receipt)

    def execute(cohort, index):
        assignment = cohort["assignment"]
        prepare_cohort(assignment)
        evaluator = ExplicitMotionSuccessEvaluator.__new__(ExplicitMotionSuccessEvaluator)
        evaluator.__dict__ = trainer.__dict__.copy()
        for key in ("_success_observation_transform", "_rendered_paired_baseline",
                    "_rendered_cohort_index", "_paired_initial_state_attestation",
                    "_require_rendered_success_observations", "_rendered_success_evaluation_binding",
                    "_continuation_initial_evidence", "_continuation_initial_state_gate",
                    '_scaled_initialization_distribution', '_scaled_initial_state_attestation'):
            evaluator.__dict__.pop(key, None)
        evaluator._require_rendered_success_observations = arm in {"bc", "dagger"}
        evaluator.__dict__.pop('_raven_pooled_benchmark', None)
        evaluator.__dict__.pop('_scaled_teacher_benchmark', None)
        if _teacher_benchmark is not None:
            evaluator._scaled_teacher_benchmark = deepcopy(_teacher_benchmark)
        if _raven_benchmark is not None:
            evaluator._raven_pooled_benchmark = deepcopy(_raven_benchmark)
        if (_scaled_partition is not None and _raven_benchmark is None) or _teacher_benchmark is not None:
            from scaled_success_contract import digest as scaled_digest
            evaluator._scaled_initialization_distribution = scaled_digest(panel['scaled_protocol'])
        if _continuation is not None and index == 73:
            evaluator._continuation_initial_evidence = evidence
        output = root / f"cohort-{index:03d}"
        adapter = None
        try:
            if rendered:
                from rendered_success_observations import GeneratedSuccessObservations, GeneratedPooledSuccessObservations
                from pipeline.dagger.services.render_student import _Image3FLoopbackClient
                config = rendered_observation_config
                endpoint = config["renderer"]
                binding_options = {}
                if not paired_rendered:
                    from rendered_student_success_contract import observation_binding
                    binding = observation_binding(panel, config, index)
                    binding_options["evaluation_binding"] = binding
                    evaluator._rendered_success_evaluation_binding = binding
                transport_options = ({"transport_recovery": TRANSPORT["schema"],
                                      "recovery_timeout_s": TRANSPORT["recovery_timeout_s"]}
                                     if _continuation is not None else {})
                if _scaled_partition is not None:
                    from rendered_endpoint_pool import SHARDED_ROUTING
                    transport_options = dict(transport_recovery='roborender.image3f.cached-reply.v1', recovery_timeout_s=75.)
                    binding_options['endpoint_pool_config'] = endpoint
                    if endpoint['routing'] == SHARDED_ROUTING:
                        binding_options['endpoint_pool_physical_gpu'] = int(os.environ['RESIDENT_PHYSICAL_GPU'])
                def make_client(lane):
                    if _scaled_partition is not None:
                        from rendered_endpoint_pool import endpoint_slot_for_lane
                        slot = endpoint_slot_for_lane(endpoint, lane, cohort_index=index,
                            physical_gpu=binding_options.get('endpoint_pool_physical_gpu'))
                        selected = endpoint['endpoints'][slot]
                    else:
                        selected = endpoint
                    return _Image3FLoopbackClient(selected['host'], selected['port'], **transport_options)
                adapter_type = GeneratedSuccessObservations
                if _raven_benchmark is not None:
                    adapter_type = GeneratedPooledSuccessObservations
                    binding_options['benchmark_sha256'] = hashlib.sha256(canonical(_raven_benchmark)).hexdigest()
                adapter = adapter_type(trainer,
                    client_factory=make_client,
                    anchor_files=config["anchors"]["files"],
                    anchor_seed=(config["anchors"]["seed"] + index) % (2 ** 32),
                    renderer_identity=endpoint["identity"],
                    preview_directory=root / "render-previews" / f"cohort-{index:03d}",
                    telemetry_path=root / "render-timings" / f"cohort-{index:03d}.jsonl",
                    **binding_options)
                evaluator._success_observation_transform = adapter
                if paired_rendered:
                    evaluator._rendered_paired_baseline = rendered_paired_baseline
                    evaluator._rendered_cohort_index = index
            from contextlib import nullcontext
            if _teacher_benchmark is not None:
                from isolated_ppo_teacher import teacher_only_execution
                teacher_context = teacher_only_execution(evaluator, trainer._scaled_teacher_private,
                                                         shared_trainer=trainer)
            else:
                teacher_context = nullcontext(None)
            with teacher_context as isolation:
                value = evaluator.evaluate_cohort(assignment, checkpoint_path=checkpoint_path,
                    checkpoint_sha256=checkpoint_sha256, output=output, arm=arm,
                    teacher_action_mode=teacher_action_mode, numeric_trace_steps=numeric_trace_steps,
                    diagnostic_initial_robot_refresh=diagnostic_initial_robot_refresh,
                    reference=str(Path(reference_root) / output.name) if reference_root else "")
            if _teacher_benchmark is not None:
                value['teacher_isolation'] = deepcopy(isolation)
                value['training_curriculum_restored'] = False
            if rendered and not paired_rendered:
                from rendered_student_success_contract import validate_cohort_observations
                validate_cohort_observations(value, count=len(assignment["lanes"]),
                    config=rendered_observation_config, cohort_index=index, panel=panel,
                    **({'_pooled_benchmark': _raven_benchmark} if _raven_benchmark is not None else {}))
        finally:
            if adapter is not None:
                adapter.close()
        if teacher_action_mode == RAVEN_PPO and _teacher_benchmark is None:
            trainer._ppo_teacher_parameter_attestation = evaluator._ppo_teacher_parameter_attestation
        return value

    if _teacher_benchmark is not None:
        from scaled_teacher_benchmark import wrap_result
        index = _teacher_benchmark['cohort_index']
        result = wrap_result(panel, _teacher_benchmark, execute(panel['cohorts'][index], index))
        publish_json(root / 'result.json', result)
        return result

    if _raven_benchmark is not None:
        from raven_pooled_benchmark import validate_result
        index = _raven_benchmark['cohort_index']
        result = execute(panel['cohorts'][index], index)
        validate_result(result, _raven_benchmark)
        publish_json(root / 'result.json', result)
        return result

    if _scaled_partition is not None:
        from explicit_motion_panel import run_partition
        from scaled_success_contract import wrap_partition
        partition_result = run_partition(panel, _scaled_partition, execute_cohort=execute,
            checkpoint_sha256=checkpoint_sha256, arm=arm, report=report,
            teacher_action_mode=teacher_action_mode,
            diagnostic_initial_robot_refresh=diagnostic_initial_robot_refresh,
            save_receipt=lambda receipt: publish_json(receipts_root / f"cohort-{receipt['cohort_index']:03d}.json", receipt))
        result = wrap_partition(panel, partition_result, checkpoint_sha256, rendered_observation_config)
        publish_json(root / 'result.json', result)
        return result

    if _continuation is not None:
        from explicit_motion_panel import make_cohort_receipt
        from rendered_success_continuation import merge_results
        new_receipts = []
        for index in range(73, len(panel["cohorts"])):
            spec = panel["cohorts"][index]
            report({"phase": "rendered_success_continuation_cohort_start", "cohort_index": index,
                "cohort_id": spec["assignment"]["cohort_id"], "case_ids": spec["case_ids"],
                "scored_lanes": [1] if index == 73 else list(range(len(spec["case_ids"]))),
                "unscored_companion_lanes": [0] if index == 73 else []})
            receipt = make_cohort_receipt(panel, index, execute(spec, index),
                checkpoint_sha256=checkpoint_sha256, arm=arm, teacher_action_mode=teacher_action_mode,
                diagnostic_initial_robot_refresh=diagnostic_initial_robot_refresh)
            new_receipts.append(receipt)
            if index == 73:
                # This also checks the before-action gate and excludes lane0.
                progress = merge_results(panel, rendered_observation_config, evidence, admission,
                                         new_receipts, require_complete=False)
            else:
                # execute/make_cohort_receipt already validate this fresh native
                # receipt. Avoid rehashing every earlier receipt at every cohort;
                # the final merge independently revalidates the whole sequence.
                count = len(receipt["case_ids"])
                successes = receipt["result"]["native_successes"]
                for name in ("completed_episodes", "strict_accepted_episodes", "new_scored_episodes"):
                    progress[name] += count
                for name in ("successes", "strict_accepted_successes"):
                    progress[name] += successes
            publish_json(receipts_root / f"cohort-{index:03d}.json", receipt)
            report({"phase": "rendered_success_continuation_progress", **{name: progress[name] for name in (
                "completed_episodes", "expected_episodes", "successes", "strict_accepted_episodes",
                "strict_accepted_successes", "weaker_evidence_episodes", "new_scored_episodes")}})
        result = merge_results(panel, rendered_observation_config, evidence, admission, new_receipts)
        publish_json(root / "result.json", result)
        return result

    result = run_panel(panel, execute_cohort=execute, report=report,
        completed_receipts=completed_receipts,
        expected_checkpoint_sha256=checkpoint_sha256, expected_arm=arm,
        teacher_action_mode=teacher_action_mode,
        diagnostic_initial_robot_refresh=diagnostic_initial_robot_refresh,
        save_receipt=lambda receipt: publish_json(
            receipts_root / f"cohort-{receipt['cohort_index']:03d}.json", receipt))
    if paired_rendered:
        from rendered_training_support_contract import protocol_identity, validate_rendered_result
        result.update(rendered_training_support_protocol=protocol_identity(panel, rendered_observation_config),
                      diagnostic_only=True, exact_historical_replay=False)
        validate_rendered_result(panel, result, checkpoint_sha256,
                                 rendered_observation_config, rendered_paired_baseline)
    elif rendered:
        from rendered_student_success_contract import protocol_identity, validate_rendered_result
        result["rendered_student_success_protocol"] = protocol_identity(panel, rendered_observation_config)
        if diagnostic_initial_robot_refresh == STUDENT_CLEAR_ONCE:
            from production_success_contract import protocol_identity as student_protocol, validate_production_result
            result["production_success_protocol"] = student_protocol(panel)
            validate_production_result(panel, result, checkpoint_sha256, allowed_arms=(arm,),
                                       rendered_config=rendered_observation_config)
        else:
            from smoke_success_contract import protocol_identity as student_protocol, validate_smoke_result
            result["smoke_success_protocol"] = student_protocol(panel)
            validate_smoke_result(panel, result, checkpoint_sha256, allowed_arms=(arm,),
                                  rendered_config=rendered_observation_config)
        validate_rendered_result(panel, result, checkpoint_sha256, rendered_observation_config)
    elif training_support_selection is not None:
        from training_support_success_contract import protocol_identity, validate_training_support_result
        result.update(training_support_success_protocol=protocol_identity(panel),
                      diagnostic_only=True, exact_historical_replay=False)
        validate_training_support_result(panel, result, checkpoint_sha256, allowed_arms=(arm,))
    elif diagnostic_initial_robot_refresh == STUDENT_CLEAR_ONCE:
        from production_success_contract import protocol_identity, validate_production_result
        result["production_success_protocol"] = protocol_identity(panel)
        validate_production_result(panel, result, checkpoint_sha256, allowed_arms=(arm,))
    elif diagnostic_initial_robot_refresh == SMOKE_STUDENT_CLEAR_ONCE:
        from smoke_success_contract import protocol_identity, validate_smoke_result
        result["smoke_success_protocol"] = protocol_identity(panel)
        validate_smoke_result(panel, result, checkpoint_sha256, allowed_arms=(arm,))
    publish_json(root / "result.json", result)
    return result
