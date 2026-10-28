"""Collection-bound online mixed-policy diagnostic (not pure-student scoring).

Reuse native teacher reset/accounting and Image3F preprocessing. Every current
lane is rendered before a frozen student forward; only the 11-D manipulation
action is mixed. One batched Isaac step follows the all-lane rendering barrier.
"""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path

from .teacher_episodes import (TeacherModeBackend, PerLaneEpisodes, digest,
                              make_spec as teacher_spec, publish_status,
                              require_full_initial_clocks, wait_pair_start)

SCHEMA = 'roborender.collection-bound-mixed-episodes.v1'
PAIR_SCHEMA = 'roborender.mixed32-pair.v1'
CHECKPOINT_SHA256 = '720a975812f6083192234d650033511c821b6902da2ef7734d87943f6c0a359a'


def make_spec(request, request_sha256, *, seed, inputs_sha256):
    spec = teacher_spec(request, request_sha256, seed=seed)
    if (type(inputs_sha256) is not str or len(inputs_sha256) != 64
            or any(c not in '0123456789abcdef' for c in inputs_sha256)):
        raise ValueError('immutable mixed inputs required')
    spec.update(schema=SCHEMA, policy='mixed_ppo_teacher_frozen_bootstrap_student',
                teacher_probability=.5, selection='independent_per_lane_per_vector_step',
                selection_seed=seed + 10000, anchor_seed=seed + 20000,
                renderer='Image3F5_generated_only', inputs_sha256=inputs_sha256,
                checkpoint_sha256=CHECKPOINT_SHA256,
                checkpoint_load_scope='policy_only_no_optimizer_environment_or_rng_restore')
    del spec['renderer_calls'], spec['student_forward_calls']
    return spec


def load_spec():
    raw = Path('/run/teacher-episodes.json').read_bytes()
    if hashlib.sha256(raw).hexdigest() != os.environ['ROBORENDER_TEACHER_EPISODES_SHA256']:
        raise RuntimeError('mixed spec changed')
    value = json.loads(raw)
    bc = Path('/run/bc-source-request.json').read_bytes()
    inputs = Path('/run/mixed-inputs.json').read_bytes()
    expected = make_spec(json.loads(bc), hashlib.sha256(bc).hexdigest(),
                        seed=value['seed'], inputs_sha256=hashlib.sha256(inputs).hexdigest())
    if value != expected:
        raise RuntimeError('mixed evaluation differs from immutable inputs/BC request')
    return value


def mix_actions(teacher26, student11, mask):
    """No hidden teacher takeover, velocity correction or changed Homie tail."""
    import torch
    count = teacher26.shape[0]
    if (tuple(teacher26.shape) != (count,26) or tuple(student11.shape) != (count,11)
            or tuple(mask.shape) != (count,) or mask.dtype != torch.bool
            or teacher26.device != student11.device or mask.device != teacher26.device
            or not torch.isfinite(teacher26).all() or not torch.isfinite(student11).all()):
        raise RuntimeError('invalid mixed action batch')
    return {'actions': torch.cat((torch.where(mask[:,None], teacher26[:,:11], student11),
                                  teacher26[:,11:]), dim=-1)}


def policy_only(checkpoint):
    """Never pass a saved trainer/curriculum state to any runtime load hook."""
    if not isinstance(checkpoint,dict) or not isinstance(checkpoint.get('policy_state_dict'),dict):
        raise ValueError('bootstrap needs explicit policy_state_dict')
    return checkpoint['policy_state_dict']


class MixedModeBackend(TeacherModeBackend):
    schema, mode, route, arm = SCHEMA, 'STUDENT_TEST', 'evaluate_collection_mixed', 'mixed'

    def valid_policy_result(self, result):
        n = result.get('vector_steps', 0)
        evidence = result.get('rendered_observation_receipt', {})
        return (type(n) is int and n > 0 and result.get('optimizer_steps') == 0
                and result.get('student_forward_calls') == n
                and result.get('renderer_calls') == n*32
                and result.get('frozen_student_unchanged') is True
                and result.get('checkpoint_sha256') == CHECKPOINT_SHA256
                and evidence.get('vector_steps_rendered') == n
                and evidence.get('render_requests_completed') == n*32
                and evidence.get('raytraced_fallbacks') == 0
                and evidence.get('failed') is False)


def _native_classes():
    import time
    import torch
    import omni.kit.app
    from omegaconf import OmegaConf
    from gr00t.rl.trl.trainer.distill_trainer_obj_pred_homie_api import TRLDistillTrainerObjPredHomieAPI
    from gr00t.rl.trl.trainer.distill_trainer_videogen import TRLDistillTrainerVideoGen
    from .teacher_episodes import _native_classes as teacher_classes
    from .policy_actions import prepare_ppo_teacher, raven_ppo_step
    from .student_observations import (GeneratedSuccessObservations,
                                      apply_observation_transform, observation_receipt)
    from pipeline.deployment.rtx_actor.initial_reset_contract import require_completed_initial_reset
    from pipeline.deployment.rtx_actor.rendered_endpoint_pool import endpoint_slot_for_lane, validate_pool_config
    from pipeline.deployment.rtx_actor.bc_resource_observer import wait_for_observer
    from pipeline.dagger.services.render_student import _Image3FLoopbackClient
    from dagger_rtx.resident_service import ResidentOwner, serve

    IsaacSim, Environment, _ = teacher_classes(load_spec)

    class MixedEpisodeEvaluator(TRLDistillTrainerObjPredHomieAPI):
        # Reuse the scientific preprocessing, without the VideoGen collector or
        # its historical raw/generated mixture and automatic render clients.
        _h5v4_control = TRLDistillTrainerVideoGen._h5v4_control
        _resize_seq = TRLDistillTrainerVideoGen._resize_seq
        _normalize_like_env = TRLDistillTrainerVideoGen._normalize_like_env

        def train(self):
            root = Path('/dagger-runtime')
            spec = load_spec()
            self.env.simulator.sim.pause()
            backend = MixedModeBackend(spec, evaluate=self.evaluate,
                pause=self.env.simulator.sim.pause,
                paused=lambda: not self.env.simulator.sim.is_playing(),
                kit_identity=lambda: id(omni.kit.app.get_app()), revalidate=load_spec)
            owner = ResidentOwner(root/'mode-owner', backend,
                gpu_uuid=os.environ['CUDA_VISIBLE_DEVICES'],
                source_sha256=os.environ['ROBORENDER_TEACHER_MODE_SOURCE_SHA256'])
            publish_status(root/'mode-ready.json', owner.status())
            return serve(owner, port=18640+int(os.environ['ROBORENDER_TEACHER_PAIR_RANK']),
                         token_file=root/'mode-token')

        def evaluate(self):
            output, spec = Path('/dagger-runtime'), load_spec()
            rank = int(os.environ['ROBORENDER_TEACHER_PAIR_RANK'])
            adapter = None
            original_optimizer, original_reset = self.optimizer.step, self.env._reset_task_states
            def emit(phase, **data):
                publish_status(output/'status.json', dict(phase=phase, pid=os.getpid(),
                    time_ns=time.time_ns(), **data))
            try:
                if (self.env.num_envs != 32 or not isinstance(self.env, Environment)
                        or self.args.num_ppo_epochs != 0 or self.state.global_step != 0):
                    raise RuntimeError('mixed diagnostic requires fresh frozen B32 evaluation')
                def forbidden(*args, **kwargs):
                    raise RuntimeError('optimization forbidden in mixed evaluation')
                self.optimizer.step = forbidden
                inputs = json.loads(Path('/run/mixed-inputs.json').read_text())
                cp = Path('/mixed-assets/bootstrap.pt')
                if hashlib.sha256(cp.read_bytes()).hexdigest() != CHECKPOINT_SHA256:
                    raise RuntimeError('bootstrap checkpoint identity mismatch')
                checkpoint = torch.load(cp, map_location='cpu', weights_only=False)
                expected = policy_only(checkpoint)
                del checkpoint  # Saved trainer/optimizer/curriculum are never restored.
                self.policy_model.load_state_dict(expected, strict=True)
                def unchanged():
                    live = self.policy_model.state_dict()
                    return set(live) == set(expected) and all(torch.equal(v.detach().cpu(), expected[k])
                                                             for k,v in live.items())
                if not unchanged():
                    raise RuntimeError('strict bootstrap loading failed')
                self._eval_mode()
                self.policy_model.eval_mode()
                self.policy_model.init_rollout()
                if self.policy_model.is_recurrent or self.policy_model.max_rollout_history != 1:
                    raise RuntimeError('bootstrap diagnostic admits only memoryless one-frame student')
                self.ref_model.eval()
                self.ref_model.init_rollout()
                teacher_proof = prepare_ppo_teacher(self, original_ppo=True)
                publish_status(output/'teacher-parameters.json', teacher_proof)
                vg = self.config.videogen
                self.vg_control_res = list(vg.control_resolution)
                self.vg_depth_mode = vg.depth_mode
                self.vg_h5v4_per_view = dict(vg.get('depth_h5v4_per_view', {}))
                self.vg_h5v4_default = dict(vg.get('depth_h5v4_default',
                    {'raw_clip':[0.,10.], 'inv_clip':[1.,20.]}))
                self.vg_invalid_fill = vg.get('invalid_fill','far')
                endpoint = validate_pool_config(inputs['renderer'])
                binding = dict(schema='roborender.rendered-student-success-cohort-binding.v1',
                    checkpoint_sha256=CHECKPOINT_SHA256, panel_sha256=spec['bc_request_sha256'],
                    config_sha256=digest(spec), cohort_index=rank,
                    motion_assignment_sha256=digest(spec['motions']))
                def client(lane):
                    slot = endpoint_slot_for_lane(endpoint, lane, cohort_index=rank, physical_gpu=rank)
                    e = endpoint['endpoints'][slot]
                    return _Image3FLoopbackClient(e['host'],e['port'],
                        transport_recovery='roborender.image3f.cached-reply.v1', recovery_timeout_s=75.)
                adapter = GeneratedSuccessObservations(self, client_factory=client,
                    anchor_files=inputs['anchors'], anchor_seed=spec['anchor_seed'],
                    renderer_identity=endpoint['identity'], evaluation_binding=binding,
                    endpoint_pool_config=endpoint, endpoint_pool_physical_gpu=rank,
                    telemetry_path=output/'render-timings.jsonl')
                self._success_observation_transform = adapter
                self._require_rendered_success_observations = True
                self._rendered_success_evaluation_binding = binding
                if list(self.env.reset_dataset.motion_files) != [m['path'] for m in spec['motion_inventory']]:
                    raise RuntimeError('full reset inventory changed')
                resets = []
                def observe_reset(lanes, *args, **kwargs):
                    value = original_reset(lanes, *args, **kwargs)
                    ranges = self.env.reset_dataset.sample_idx_based_on_demo_clip_number
                    for lane,sample,index in zip(lanes.tolist(),self.env.sample_idx.tolist(),
                                                 self.env.sample_idx_in_seq.tolist()):
                        if not int(ranges[lane][0]) <= sample <= int(ranges[lane][1]):
                            raise RuntimeError('reset left admitted BC motion support')
                        resets.append(dict(lane=lane,sample_idx=sample,sample_idx_in_seq=index))
                    return value
                self.env._reset_task_states = observe_reset
                self.env.simulator.sim.play()
                obs = self.env.reset_all()
                publish_status(output/'initial-reset.json', dict(reset=require_completed_initial_reset(self.env),
                    clocks=require_full_initial_clocks(self.env),
                    motion_binding=self.env.simulator.teacher_motion_binding, resets=deepcopy(resets)))
                OmegaConf.save(self.env.config, output/'environment-config.yaml', resolve=True)
                OmegaConf.save(self.config, output/'policy-config.yaml', resolve=True)
                wait_pair_start(output, spec, emit)
                account = PerLaneEpisodes(32,2)
                dones = torch.zeros(32, device=self.accelerator.device)
                selector = torch.Generator(device='cpu').manual_seed(spec['selection_seed'])
                teacher_counts = torch.zeros(32,dtype=torch.int64)
                started = time.monotonic()
                with torch.no_grad(), (output/'transitions.jsonl').open('x') as stream:
                    for step in range(spec['max_vector_steps']):
                        wait_for_observer(output)
                        emit('RENDERING', vector_steps=step, completed=len(account.outcomes))
                        obs = {key:value.to(self.accelerator.device) for key,value in obs.items()}
                        generated = apply_observation_transform(self, obs, dones, step)
                        # Student rollout updates its own history only. Teacher
                        # and Homie execute exactly once on unchanged native obs.
                        student = self.policy_model.rollout_with_obj_pred(obs_dict=generated,
                            episode_attnmask=None, cur_dones=dones)['action_mean']
                        teacher, _ = raven_ppo_step(self, obs)
                        mask_cpu = torch.rand(32,generator=selector) < .5
                        teacher_counts += mask_cpu
                        actions = mix_actions(teacher['actions'],student,mask_cpu.to(student.device))
                        action_trace = dict(teacher26=teacher['actions'].detach().cpu().tolist(),
                            student11=student.detach().cpu().tolist(),
                            selected26=actions['actions'].detach().cpu().tolist())
                        obs, reward, dones, infos = self.env.step(actions)
                        goal = infos['grasp_goal_reached']
                        if not torch.equal(goal,self.env._transition_goal_before_reset):
                            raise RuntimeError('pre-reset goal accounting disagreement')
                        rows = account.update(step,goal.bool().tolist(),dones.bool().tolist(),
                                              infos['time_outs'].bool().tolist())
                        stream.write(json.dumps(dict(step=step,completed=rows,resets=resets,
                            teacher_mask=mask_cpu.tolist(), **action_trace))+'\n')
                        resets.clear()
                        stream.flush(); os.fsync(stream.fileno())
                        emit('EVALUATING', vector_steps=step+1, teacher_decisions=int(teacher_counts.sum()),
                             **account.summary())
                        if account.complete:
                            break
                if not account.complete:
                    raise RuntimeError('safety ceiling reached: uncompleted episodes remain unscored')
                if not unchanged():
                    raise RuntimeError('student parameters/buffers changed during frozen evaluation')
                final_teacher = prepare_ppo_teacher(self, original_ppo=True)
                torch.cuda.synchronize(self.accelerator.device)
                self.env.simulator.sim.pause()
                result = dict(schema=SCHEMA, spec=spec, spec_sha256=digest(spec), **account.summary(),
                    vector_steps=step+1, seconds=time.monotonic()-started, optimizer_steps=0,
                    student_forward_calls=step+1, renderer_calls=(step+1)*32,
                    checkpoint_sha256=CHECKPOINT_SHA256,frozen_student_unchanged=True,
                    teacher_parameter_attestation=final_teacher,
                    teacher_decisions_per_lane=teacher_counts.tolist(), physics_idle=True,
                    **observation_receipt(self,step+1))
                publish_status(output/'result.json',result)
                emit('COMPLETE_IDLE',**account.summary())
                return result
            except BaseException as exc:
                self.env.simulator.sim.pause()
                emit('FAILED',error_type=type(exc).__name__,error=str(exc))
                raise
            finally:
                self.env._reset_task_states = original_reset
                self.optimizer.step = original_optimizer
                if adapter is not None:
                    adapter.close()
                self._success_observation_transform = None

    return IsaacSim, Environment, MixedEpisodeEvaluator


def __getattr__(name):
    names = ('IsaacSim','Environment','MixedEpisodeEvaluator')
    if name not in names:
        raise AttributeError(name)
    globals().update(zip(names,_native_classes()))
    return globals()[name]
