"""Focused CPU-only tests for the additive round-DAgger runtime facade."""

from __future__ import annotations

import random
import unittest

from round_dagger.runtime import (
    ActionDecision,
    ArrayMetadata,
    DaggerAgentAdapter,
    DecisionRecord,
    RoboRenderObservationTransform,
    RoundContext,
    RuntimeAdapterError,
    RuntimeSubscriber,
    ShadowRuntimeBridge,
    StepRecord,
    SubscriberBus,
    SubscriberExecutionError,
    SubscriberSideEffectError,
    VectorEnvironmentAdapter,
    runtime_adapter_mode,
)


class FakeTensor:
    """Small tensor double exposing array metadata."""

    def __init__(self, label: str, shape: tuple[int, ...] = (2, 3)):
        self.label = label
        self.shape = shape
        self.dtype = "float32"
        self.device = "cpu"

def policy_state(prefix: str = "action") -> dict[str, FakeTensor]:
    return {
        "student_actions": FakeTensor(f"{prefix}-student"),
        "gt_actions": FakeTensor(f"{prefix}-expert"),
        "actions": FakeTensor(f"{prefix}-behavior"),
        "expert_mask": FakeTensor(f"{prefix}-mask", (2, 1)),
    }


class FakeStorage:
    def __init__(self, rows: dict[str, list[object]], *, step: int):
        self.rows = rows
        self.step = step

    def query_key(self, key: str) -> list[object]:
        return self.rows[key]


def storage_rows() -> tuple[FakeStorage, dict[str, list[object]]]:
    rows: dict[str, list[object]] = {
        "vision_obs": [FakeTensor("vision-0", (2, 4)), FakeTensor("vision-1", (2, 4))],
        "actor_obs": [FakeTensor("actor-0", (2, 5)), FakeTensor("actor-1", (2, 5))],
        "student_actions": [FakeTensor("student-0"), FakeTensor("student-1")],
        "gt_actions": [FakeTensor("expert-0"), FakeTensor("expert-1")],
        "actions": [FakeTensor("behavior-0"), FakeTensor("behavior-1")],
        "expert_mask": [FakeTensor("mask-0", (2, 1)), FakeTensor("mask-1", (2, 1))],
        "rewards": [FakeTensor("rewards-0", (2, 1)), FakeTensor("rewards-1", (2, 1))],
        "dones": [FakeTensor("dones-0", (2, 1)), FakeTensor("dones-1", (2, 1))],
        "time_outs": [FakeTensor("timeouts-0", (2, 1)), FakeTensor("timeouts-1", (2, 1))],
        "round_global_env_ids": [FakeTensor("env-ids-0", (2, 1)), FakeTensor("env-ids-1", (2, 1))],
        "round_episode_ids": [
            FakeTensor("episode-ids-0", (2, 1)),
            FakeTensor("episode-ids-1", (2, 1)),
        ],
        "round_episode_steps": [
            FakeTensor("episode-steps-0", (2, 1)),
            FakeTensor("episode-steps-1", (2, 1)),
        ],
        "round_observation_is_raytraced": [
            FakeTensor("raytraced-0", (2, 1)),
            FakeTensor("raytraced-1", (2, 1)),
        ],
    }
    return FakeStorage(rows, step=2), rows


class RuntimeModeTest(unittest.TestCase):
    def test_default_is_legacy_and_invalid_values_fail_closed(self) -> None:
        self.assertEqual(runtime_adapter_mode({}), "legacy")
        self.assertEqual(
            runtime_adapter_mode(
                {
                    "ROUND_DAGGER_RUNTIME_ADAPTER_MODE": " SHADOW ",
                    "ROUND_DAGGER_RUNTIME_ADAPTER_SHADOW_ACK": "metadata-only-v1",
                }
            ),
            "shadow",
        )
        with self.assertRaisesRegex(RuntimeAdapterError, "must be one of legacy, shadow"):
            runtime_adapter_mode({"ROUND_DAGGER_RUNTIME_ADAPTER_MODE": "active"})
        with self.assertRaisesRegex(RuntimeAdapterError, "shadow mode requires"):
            runtime_adapter_mode(
                {"ROUND_DAGGER_RUNTIME_ADAPTER_MODE": "shadow"}
            )


class VectorEnvironmentAdapterTest(unittest.TestCase):
    def test_exact_pass_through_and_done_never_triggers_an_extra_reset(self) -> None:
        reset_observations = {"actor_obs": object()}
        next_observations = {"actor_obs": object()}
        rewards = object()
        dones = [True, False]
        infos = {"time_outs": object()}
        actor_state = {"actions": object(), "gt_actions": object()}

        class Environment:
            num_envs = 2

            def __init__(self) -> None:
                self.reset_calls = 0
                self.partial_reset_calls = 0
                self.step_calls = 0
                self.received_actor_state = None

            def reset_all(self):
                self.reset_calls += 1
                return reset_observations

            def reset_envs_idx(self, _env_ids):
                self.partial_reset_calls += 1

            def step(self, value):
                self.step_calls += 1
                self.received_actor_state = value
                return next_observations, rewards, dones, infos

        environment = Environment()
        adapter = VectorEnvironmentAdapter(environment)
        self.assertEqual(adapter.num_envs, 2)
        self.assertIs(adapter.reset_all(), reset_observations)

        result = adapter.step(actor_state)

        self.assertIs(environment.received_actor_state, actor_state)
        self.assertIs(result.observations, next_observations)
        self.assertIs(result.rewards, rewards)
        self.assertIs(result.dones, dones)
        self.assertIs(result.infos, infos)
        self.assertEqual(environment.step_calls, 1)
        self.assertEqual(environment.reset_calls, 1)
        self.assertEqual(environment.partial_reset_calls, 0)


class ObservationTransformTest(unittest.TestCase):
    def test_relabel_runs_once_and_replaces_rgb_in_the_original_mapping(self) -> None:
        simulated = FakeTensor("simulated", (2, 12, 8, 3))
        generated = FakeTensor("generated", (2, 12, 8, 3))
        observations = {"vision_obs": simulated, "actor_obs": object()}
        dones = object()
        calls: list[tuple[object, object]] = []

        def relabel(value, cur_dones):
            calls.append((value, cur_dones))
            return {"vision_obs": generated}

        returned = RoboRenderObservationTransform(relabel).apply(observations, dones)

        self.assertIs(returned, observations)
        self.assertIs(observations["vision_obs"], generated)
        self.assertEqual(calls, [(observations, dones)])

    def test_shape_mismatch_is_rejected_without_installing_the_output(self) -> None:
        simulated = FakeTensor("simulated", (2, 12, 8, 3))
        generated = FakeTensor("generated", (2, 11, 8, 3))
        observations = {"vision_obs": simulated}
        transform = RoboRenderObservationTransform(
            lambda _observations, _dones: {"vision_obs": generated}
        )

        with self.assertRaisesRegex(RuntimeAdapterError, "changed policy observation shape"):
            transform.apply(observations)
        self.assertIs(observations["vision_obs"], simulated)


class DaggerAgentAdapterTest(unittest.TestCase):
    def test_policy_is_called_once_without_consuming_rng_and_actions_map_exactly(self) -> None:
        state = policy_state()
        observations = {"vision_obs": object()}
        dones = object()
        calls: list[tuple[object, object]] = []

        def policy_step(value, cur_dones):
            calls.append((value, cur_dones))
            return state

        random.seed(173)
        rng_before = random.getstate()
        decision = DaggerAgentAdapter(policy_step, beta=0.35).decide(
            observations, dones
        )
        rng_after = random.getstate()

        self.assertEqual(calls, [(observations, dones)])
        self.assertEqual(rng_before, rng_after)
        self.assertIs(decision.student_actions, state["student_actions"])
        self.assertIs(decision.expert_actions, state["gt_actions"])
        self.assertIs(decision.behavior_actions, state["actions"])
        self.assertIs(decision.expert_mask, state["expert_mask"])
        self.assertEqual(decision.beta, 0.35)
        with self.assertRaises(TypeError):
            decision.policy_state["actions"] = object()  # type: ignore[index]

    def test_adapter_does_not_advance_torch_rng(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch is unavailable")
        state = {
            "student_actions": torch.zeros(2, 3),
            "gt_actions": torch.ones(2, 3),
            "actions": torch.zeros(2, 3),
            "expert_mask": torch.zeros(2, 1, dtype=torch.bool),
        }
        torch.manual_seed(913)
        before = torch.random.get_rng_state().clone()
        DaggerAgentAdapter(lambda _obs, _dones: state, beta=0.5).decide({})
        after = torch.random.get_rng_state()
        self.assertTrue(torch.equal(before, after))


class SubscriberBusTest(unittest.TestCase):
    def test_subscribers_run_in_registration_order(self) -> None:
        events: list[str] = []

        class Recorder(RuntimeSubscriber):
            def __init__(self, name: str):
                self.name = name

            def on_round_start(self, _context):
                events.append(self.name)

        bus = SubscriberBus([Recorder("first"), Recorder("second")])
        bus.publish_round_start(RoundContext.create(num_envs=2, expected_steps=1))
        self.assertEqual(events, ["first", "second"])
        with self.assertRaisesRegex(RuntimeAdapterError, "cannot be added"):
            bus.subscribe(Recorder("late"))

    def test_unmarked_callback_object_is_rejected(self) -> None:
        class Unmarked:
            def on_round_start(self, _context):
                pass

        with self.assertRaisesRegex(RuntimeAdapterError, "explicitly subclass"):
            SubscriberBus([Unmarked()])  # type: ignore[list-item]

    def test_subscriber_failure_is_wrapped_and_later_subscribers_do_not_run(self) -> None:
        events: list[str] = []

        class Failing(RuntimeSubscriber):
            def on_round_start(self, _context):
                events.append("failing")
                raise ValueError("injected failure")

        class Later(RuntimeSubscriber):
            def on_round_start(self, _context):
                events.append("later")

        bus = SubscriberBus([Failing(), Later()])
        with self.assertRaises(SubscriberExecutionError) as caught:
            bus.publish_round_start(RoundContext.create(num_envs=2, expected_steps=1))
        self.assertIsInstance(caught.exception.__cause__, ValueError)
        self.assertEqual(events, ["failing"])

    def test_default_subscriber_payload_contains_metadata_not_tensor_values(self) -> None:
        state = policy_state()
        record = DecisionRecord(
            rollout_step=0,
            observations={"vision_obs": FakeTensor("vision", (2, 4))},
            decision=ActionDecision.from_policy_state(state, beta=0.5),
        )
        received = []

        class Inspecting(RuntimeSubscriber):
            def on_decision(self, value):
                received.append(value)

        SubscriberBus([Inspecting()]).publish_decision(record)
        self.assertEqual(len(received), 1)
        projected = received[0]
        self.assertIsInstance(projected.observations["vision_obs"], ArrayMetadata)
        self.assertIsInstance(projected.decision.behavior_actions, ArrayMetadata)
        self.assertEqual(projected.decision.behavior_actions.shape, (2, 3))
        self.assertFalse(hasattr(projected.decision.behavior_actions, "mutate"))

    def test_global_rng_side_effect_is_restored_and_rejected(self) -> None:
        try:
            import numpy as np
            import torch
        except ImportError:
            self.skipTest("NumPy or PyTorch is unavailable")

        random.seed(421)
        np.random.seed(421)
        torch.manual_seed(421)
        python_before = random.getstate()
        numpy_before = np.random.get_state()
        torch_before = torch.random.get_rng_state().clone()

        class Consuming(RuntimeSubscriber):
            def on_round_start(self, _context):
                random.random()
                np.random.random()
                torch.rand(1)

        with self.assertRaisesRegex(
            SubscriberSideEffectError, "changed actor RNG state"
        ):
            SubscriberBus([Consuming()]).publish_round_start(
                RoundContext.create(num_envs=2, expected_steps=1)
            )

        self.assertEqual(random.getstate(), python_before)
        numpy_after = np.random.get_state()
        self.assertEqual(numpy_after[0], numpy_before[0])
        np.testing.assert_array_equal(numpy_after[1], numpy_before[1])
        self.assertEqual(numpy_after[2:], numpy_before[2:])
        self.assertTrue(torch.equal(torch.random.get_rng_state(), torch_before))


class CanonicalStorageRecordTest(unittest.TestCase):
    def test_record_uses_the_requested_canonical_storage_row(self) -> None:
        storage, rows = storage_rows()
        infos = {"episode": {"success": 1}}

        record = StepRecord.from_storage(
            storage,
            rollout_step=1,
            observation_keys=("vision_obs", "actor_obs"),
            beta=0.25,
            infos=infos,
        )

        self.assertEqual(record.rollout_step, 1)
        self.assertIs(record.observations["vision_obs"], rows["vision_obs"][1])
        self.assertIs(record.observations["actor_obs"], rows["actor_obs"][1])
        self.assertIs(record.decision.student_actions, rows["student_actions"][1])
        self.assertIs(record.decision.expert_actions, rows["gt_actions"][1])
        self.assertIs(record.decision.behavior_actions, rows["actions"][1])
        self.assertIs(record.decision.expert_mask, rows["expert_mask"][1])
        self.assertIs(record.rewards, rows["rewards"][1])
        self.assertIs(record.dones, rows["dones"][1])
        self.assertIs(record.time_outs, rows["time_outs"][1])
        self.assertIs(record.global_env_ids, rows["round_global_env_ids"][1])
        self.assertIs(record.episode_ids, rows["round_episode_ids"][1])
        self.assertIs(record.episode_steps, rows["round_episode_steps"][1])
        self.assertIs(
            record.observation_is_raytraced,
            rows["round_observation_is_raytraced"][1],
        )
        self.assertEqual(record.infos["episode"], {"success": 1})
        with self.assertRaises(TypeError):
            record.observations["vision_obs"] = object()  # type: ignore[index]
        with self.assertRaises(TypeError):
            record.infos["episode"] = object()  # type: ignore[index]


class ShadowRuntimeBridgeTest(unittest.TestCase):
    def test_round_start_failure_rolls_back_bridge_state(self) -> None:
        class Failing(RuntimeSubscriber):
            def on_round_start(self, _context):
                raise ValueError("start failed")

        bridge = ShadowRuntimeBridge([Failing()])
        with self.assertRaises(SubscriberExecutionError):
            bridge.start_round(
                RoundContext.create(num_envs=2, expected_steps=1)
            )
        self.assertFalse(bridge.round_active)
        self.assertFalse(bridge.has_pending_decision)
        self.assertEqual(bridge.transition_count, 0)

    def test_state_machine_publishes_steps_in_exact_order(self) -> None:
        events: list[tuple[str, int | None]] = []

        class Recorder(RuntimeSubscriber):
            def on_round_start(self, _context):
                events.append(("start", None))

            def on_decision(self, record):
                events.append(("decision", record.rollout_step))

            def on_transition(self, record):
                events.append(("transition", record.rollout_step))

            def on_round_end(self, _context):
                events.append(("end", None))

        bridge = ShadowRuntimeBridge([Recorder()])
        context = RoundContext.create(num_envs=2, expected_steps=1)
        storage, rows = storage_rows()
        storage.step = 1
        current_policy_state = {
            key: rows[key][0]
            for key in (
                "student_actions",
                "gt_actions",
                "actions",
                "expert_mask",
                "round_global_env_ids",
                "round_episode_ids",
                "round_episode_steps",
                "round_observation_is_raytraced",
            )
        }

        bridge.start_round(context)
        bridge.begin_decision(
            rollout_step=0,
            observations={"vision_obs": rows["vision_obs"][0]},
            policy_state=current_policy_state,
            beta=0.5,
        )
        record = bridge.complete_transition(
            storage,
            observation_keys=("vision_obs", "actor_obs"),
            infos={"episode": {}},
        )
        self.assertEqual(events, [("start", None), ("decision", 0), ("transition", 0)])

        # Dones remain in the canonical record.  The phase-one bridge does not
        # introduce a second reset callback or touch Isaac reset behavior.
        self.assertIs(record.dones, rows["dones"][0])
        bridge.end_round()

        self.assertEqual(
            events,
            [
                ("start", None),
                ("decision", 0),
                ("transition", 0),
                ("end", None),
            ],
        )
        self.assertFalse(bridge.has_pending_decision)
        self.assertEqual(bridge.transition_count, 1)

    def test_live_decision_storage_mismatch_fails_closed(self) -> None:
        bridge = ShadowRuntimeBridge()
        storage, rows = storage_rows()
        storage.step = 1
        state = {
            key: rows[key][0]
            for key in (
                "student_actions",
                "gt_actions",
                "actions",
                "expert_mask",
                "round_global_env_ids",
                "round_episode_ids",
                "round_episode_steps",
                "round_observation_is_raytraced",
            )
        }
        state["actions"] = FakeTensor("different-behavior")
        bridge.start_round(RoundContext.create(num_envs=2, expected_steps=1))
        bridge.begin_decision(
            rollout_step=0,
            observations={"vision_obs": rows["vision_obs"][0]},
            policy_state=state,
            beta=0.5,
        )
        with self.assertRaisesRegex(
            RuntimeAdapterError, "differs from committed storage: actions"
        ):
            bridge.complete_transition(
                storage,
                observation_keys=("vision_obs", "actor_obs"),
                infos={"episode": {}},
            )
        self.assertTrue(bridge.has_pending_decision)
        self.assertEqual(bridge.transition_count, 0)

    def test_copied_torch_lineage_matches_committed_storage_exactly(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch is unavailable")
        live = {
            "student_actions": torch.arange(6, dtype=torch.float32).reshape(2, 3),
            "gt_actions": torch.ones(2, 3),
            "actions": torch.arange(6, dtype=torch.float32).reshape(2, 3),
            "expert_mask": torch.zeros(2, 1, dtype=torch.bool),
            "round_global_env_ids": torch.tensor([[10], [11]]),
            "round_episode_ids": torch.tensor([[2], [3]]),
            "round_episode_steps": torch.tensor([[7], [8]]),
            "round_observation_is_raytraced": torch.zeros(2, 1, dtype=torch.bool),
        }
        rows = {key: [value.clone()] for key, value in live.items()}
        rows.update(
            {
                "vision_obs": [torch.zeros(2, 4)],
                "actor_obs": [torch.zeros(2, 5)],
                "rewards": [torch.zeros(2, 1)],
                "dones": [torch.zeros(2, 1, dtype=torch.bool)],
                "time_outs": [torch.zeros(2, 1, dtype=torch.bool)],
            }
        )
        storage = FakeStorage(rows, step=1)
        bridge = ShadowRuntimeBridge()
        bridge.start_round(RoundContext.create(num_envs=2, expected_steps=1))
        bridge.begin_decision(
            rollout_step=0,
            observations={"vision_obs": rows["vision_obs"][0]},
            policy_state=live,
            beta=0.0,
        )
        record = bridge.complete_transition(
            storage,
            observation_keys=("vision_obs", "actor_obs"),
            infos={"episode": {}},
        )
        self.assertTrue(torch.equal(record.decision.behavior_actions, live["actions"]))
        bridge.end_round()

    def test_out_of_order_calls_fail_closed_without_advancing_state(self) -> None:
        bridge = ShadowRuntimeBridge()
        state = policy_state()
        with self.assertRaisesRegex(RuntimeAdapterError, "outside an active round"):
            bridge.begin_decision(
                rollout_step=0,
                observations={},
                policy_state=state,
                beta=0.5,
            )

        bridge.start_round(RoundContext.create(num_envs=2, expected_steps=1))
        with self.assertRaisesRegex(RuntimeAdapterError, "no pending decision"):
            bridge.complete_transition(
                FakeStorage({}, step=0),
                observation_keys=(),
                infos={},
            )
        bridge.begin_decision(
            rollout_step=0,
            observations={},
            policy_state=state,
            beta=0.5,
        )
        with self.assertRaisesRegex(RuntimeAdapterError, "no committed transition"):
            bridge.begin_decision(
                rollout_step=0,
                observations={},
                policy_state=state,
                beta=0.5,
            )
        with self.assertRaisesRegex(RuntimeAdapterError, "ended with a pending decision"):
            bridge.end_round()
        self.assertTrue(bridge.has_pending_decision)
        self.assertEqual(bridge.transition_count, 0)
        bridge.abort_round()
        self.assertFalse(bridge.has_pending_decision)


if __name__ == "__main__":
    unittest.main()
