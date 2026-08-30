"""Read-only student-quality bridge metric tests."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from round_dagger import artifacts
from scripts.round_dagger import student_quality_wandb_bridge as bridge
from scripts.round_dagger.student_quality_wandb_bridge import summarize_cohort
from tests.test_round_dagger_quality import DIGESTS, IDENTITY, _arrays
from tests.test_round_dagger_state import _campaign


class _FakeRun:
    def __init__(
        self,
        *,
        fail_finish: bool = False,
        fail_quality_log: bool = False,
    ) -> None:
        self.logs: list[tuple[dict, int | None]] = []
        self.summary: dict[str, object] = {}
        self.finished: list[int] = []
        self.fail_finish = fail_finish
        self.fail_quality_log = fail_quality_log

    def define_metric(self, *_args, **_kwargs) -> None:
        pass

    def log(self, payload, step=None) -> None:
        if step is not None:
            raise AssertionError("quality bridge must not use W&B's global step")
        if self.fail_quality_log and "rollout_quality/event_index" in payload:
            raise RuntimeError("simulated W&B log outage")
        self.logs.append((dict(payload), step))

    def finish(self, exit_code=0) -> None:
        self.finished.append(exit_code)
        if self.fail_finish:
            raise RuntimeError("simulated W&B sender drain failure")


class _FakeWandb:
    def __init__(self, run: _FakeRun | None = None) -> None:
        self.init_calls: list[dict] = []
        self.runs: list[_FakeRun] = []
        self._run = run

    def init(self, **kwargs):
        self.init_calls.append(kwargs)
        run = self._run or _FakeRun()
        self.runs.append(run)
        return run


class StudentQualityWandbBridgeTest(unittest.TestCase):
    def test_deep_verified_metrics_match_the_sealed_quality_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            cohort = Path(temporary).resolve() / "cohort"
            artifacts.create_cohort(
                cohort,
                expected_ranks=[0],
                expected_global_env_ids=[4, 5],
                expected_transitions=6,
                view_contract={
                    "dtype": "uint8",
                    "layout": "vertical",
                    "shape": [6, 4, 3],
                    "views": ["head_color", "hand_left", "hand_right"],
                },
                action_contract={"dtype": "float32", "shape": [13]},
                **IDENTITY,
                **DIGESTS,
            )
            artifacts.publish_rollout_shard(
                cohort,
                rank=0,
                rollout_id=0,
                part_index=0,
                arrays=_arrays(),
                **IDENTITY,
                **DIGESTS,
            )
            seal = artifacts.seal_cohort(cohort)

            record = summarize_cohort(
                cohort,
                macro_round=IDENTITY["macro_round"],
                cohort_index=IDENTITY["cohort_index"],
            )
            self.assertEqual(record["seal_sha256"], seal["cohort_sha256"])
            self.assertEqual(record["event"], 203)
            self.assertAlmostEqual(
                record["metrics"]["rollout_quality/action_mse"], 55.0 / 6.0
            )
            self.assertAlmostEqual(
                record["metrics"]["rollout_quality/expert_intervention_rate"],
                0.5,
            )
            self.assertEqual(
                record["metrics"]["rollout_quality/transitions"], 6
            )

    def test_eager_run_shows_no_data_and_driver_failure_before_any_seal(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = _campaign(Path(temporary) / "fixture")
            state_dir = Path(temporary) / "quality-observer"
            driver = Path(temporary) / "driver_status.json"
            driver.write_text(
                json.dumps(
                    {
                        "schema": "fast_roborender.round_dagger.production_driver_status.v1",
                        "campaign_root": str(root),
                        "state": "FAILED",
                        "updated_unix_ns": 123,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            fake = _FakeWandb()
            identity = SimpleNamespace(entity="nvidia", email="ldu@nvidia.com")
            arguments = [
                "--campaign-root",
                str(root),
                "--state-dir",
                str(state_dir),
                "--project",
                "roborender-round-dagger",
                "--entity",
                "nvidia",
                "--expected-email",
                "ldu@nvidia.com",
                "--authorization-scope",
                "team",
                "--driver-status",
                str(driver),
            ]
            with mock.patch.object(bridge, "_load_wandb", return_value=fake), mock.patch.object(
                bridge, "query_identity", return_value=identity
            ):
                self.assertEqual(bridge.main(arguments), 0)

            self.assertEqual(len(fake.init_calls), 1, "empty campaigns must initialize W&B")
            run = fake.runs[0]
            self.assertEqual(run.finished, [0])
            campaign_log = next(
                payload
                for payload, _step in run.logs
                if "campaign/data_ready" in payload
            )
            self.assertEqual(campaign_log["campaign/data_ready"], 0)
            self.assertEqual(campaign_log["campaign/quality_available"], 0)
            self.assertEqual(campaign_log["campaign/failed"], 1)
            self.assertEqual(campaign_log["campaign/sealed_transitions_total"], 0)
            self.assertEqual(run.summary["campaign/status"], "FAILED")
            self.assertEqual(run.summary["campaign/failure_reason"], "launcher_nonzero")
            self.assertEqual(
                run.summary["campaign/data_status"], "waiting_for_sealed_rollout"
            )

    def test_retry_status_is_bounded_and_does_not_copy_arbitrary_detail(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = _campaign(Path(temporary) / "fixture")
            campaign = artifacts.load_strict_json(root / "campaign.json")
            driver = Path(temporary) / "driver_status.json"
            secret_like_detail = "operator message must not reach telemetry"
            driver.write_text(
                json.dumps(
                    {
                        "schema": "fast_roborender.round_dagger.production_driver_status.v1",
                        "campaign_root": str(root),
                        "state": "RETRYING",
                        "updated_unix_ns": 456,
                        "collection_retry_count": 1,
                        "collection_retry_limit": 2,
                        "collection_recovery": {"safe_to_retry": True},
                        "detail": secret_like_detail,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            metrics, summary = bridge.campaign_status_projection(
                root=root,
                campaign=campaign,
                seals=[],
                driver_status_path=driver,
            )
            self.assertEqual(metrics["campaign/retrying"], 1)
            self.assertEqual(metrics["campaign/collection_retry_count"], 1)
            self.assertEqual(metrics["campaign/collection_retry_limit"], 2)
            self.assertEqual(metrics["campaign/collection_recovery_safe"], 1)
            self.assertEqual(summary["campaign/status"], "RETRYING")
            self.assertEqual(
                summary["campaign/failure_reason"], "bounded_collection_retry"
            )
            self.assertNotIn(secret_like_detail, json.dumps([metrics, summary]))

    def test_historical_seals_do_not_mark_new_round_data_ready(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = _campaign(Path(temporary) / "fixture")
            campaign = artifacts.load_strict_json(root / "campaign.json")
            historical = Path(temporary) / "macro0-seal.json"
            historical.write_text("{}\n", encoding="utf-8")
            coordinator = {
                "phase": "COLLECTING",
                "next_macro_round": 1,
                "active_policy": {"version": 1},
            }
            with mock.patch.object(
                bridge.campaign_state, "load_state", return_value=coordinator
            ):
                metrics, summary = bridge.campaign_status_projection(
                    root=root,
                    campaign=campaign,
                    seals=[(0, 0, historical)],
                    driver_status_path=None,
                )
            self.assertEqual(metrics["campaign/cohorts_sealed_total"], 1)
            self.assertEqual(metrics["campaign/current_round_cohorts_sealed"], 0)
            self.assertEqual(metrics["campaign/data_ready"], 0)
            self.assertEqual(metrics["campaign/quality_available"], 0)
            self.assertEqual(summary["campaign/data_ready"], 0)

    def test_complete_uses_last_completed_round_for_quality_readiness(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = _campaign(Path(temporary) / "fixture")
            campaign = artifacts.load_strict_json(root / "campaign.json")
            driver = Path(temporary) / "driver_status.json"
            driver.write_text(
                json.dumps(
                    {
                        "schema": "fast_roborender.round_dagger.production_driver_status.v1",
                        "campaign_root": str(root),
                        "state": "COMPLETE",
                        "updated_unix_ns": 789,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            completed = Path(temporary) / "macro4-seal.json"
            completed.write_text("{}\n", encoding="utf-8")
            coordinator = {
                "phase": "EVALUATED",
                "next_macro_round": 5,
                "active_policy": {"version": 5},
            }
            with mock.patch.object(
                bridge.campaign_state, "load_state", return_value=coordinator
            ):
                metrics, summary = bridge.campaign_status_projection(
                    root=root,
                    campaign=campaign,
                    seals=[(4, 0, completed)],
                    driver_status_path=driver,
                )
            self.assertEqual(metrics["campaign/macro_round"], 4)
            self.assertEqual(metrics["campaign/coordinator_next_macro_round"], 5)
            self.assertEqual(metrics["campaign/current_round_cohorts_sealed"], 1)
            self.assertEqual(metrics["campaign/data_ready"], 1)
            self.assertEqual(summary["campaign/data_ready"], 1)

    def test_polling_run_reports_campaign_failure_but_observer_finishes_cleanly(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = _campaign(Path(temporary) / "fixture")
            state_dir = Path(temporary) / "quality-observer"
            driver = Path(temporary) / "driver_status.json"
            driver.write_text("{}\n", encoding="utf-8")
            fake = _FakeWandb()
            identity = SimpleNamespace(entity="nvidia", email="ldu@nvidia.com")
            arguments = [
                "--campaign-root",
                str(root),
                "--state-dir",
                str(state_dir),
                "--project",
                "roborender-round-dagger",
                "--entity",
                "nvidia",
                "--expected-email",
                "ldu@nvidia.com",
                "--authorization-scope",
                "team",
                "--driver-status",
                str(driver),
                "--poll-seconds",
                "0.01",
            ]
            active = {
                "state": "LAUNCHING",
                "updated_unix_ns": 1,
                "collection_retry_count": 0,
                "collection_retry_limit": 2,
            }
            failed = {
                "state": "FAILED",
                "updated_unix_ns": 2,
                "collection_retry_count": 2,
                "collection_retry_limit": 2,
                "collection_recovery": {"safe_to_retry": True},
            }
            with mock.patch.object(
                bridge, "_load_wandb", return_value=fake
            ), mock.patch.object(
                bridge, "query_identity", return_value=identity
            ), mock.patch.object(
                bridge, "_driver_status", side_effect=[active, failed]
            ), mock.patch.object(bridge.time, "sleep", return_value=None):
                self.assertEqual(bridge.main(arguments), 0)

            self.assertEqual(len(fake.init_calls), 1)
            self.assertEqual(len(fake.runs), 1)
            self.assertEqual(fake.runs[0].finished, [0])
            self.assertEqual(len(fake.runs[0].logs), 2)
            self.assertEqual(fake.runs[0].logs[0][0]["campaign/failed"], 0)
            self.assertEqual(fake.runs[0].logs[1][0]["campaign/failed"], 1)

    def test_main_backfills_every_unseen_historical_and_current_seal(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = _campaign(Path(temporary) / "fixture")
            state_dir = Path(temporary) / "quality-observer"
            driver = Path(temporary) / "driver_status.json"
            driver.write_text(
                json.dumps(
                    {
                        "schema": "fast_roborender.round_dagger.production_driver_status.v1",
                        "campaign_root": str(root),
                        "state": "LAUNCHING",
                        "updated_unix_ns": 1,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            historical = Path(temporary) / "historical.json"
            current = Path(temporary) / "current.json"
            historical.write_text("historical\n", encoding="utf-8")
            current.write_text("current\n", encoding="utf-8")
            fake = _FakeWandb()
            identity = SimpleNamespace(entity="nvidia", email="ldu@nvidia.com")
            coordinator = {
                "phase": "COLLECTING",
                "next_macro_round": 1,
                "active_policy": {"version": 1},
            }
            historical_record = {
                "event": 0,
                "seal_sha256": bridge.sha256_file(historical),
                "metrics": {
                    "rollout_quality/event_index": 0,
                    "rollout_quality/action_mse": 0.5,
                },
            }
            current_record = {
                "event": 100,
                "seal_sha256": bridge.sha256_file(current),
                "metrics": {
                    "rollout_quality/event_index": 100,
                    "rollout_quality/action_mse": 0.25,
                },
            }
            arguments = [
                "--campaign-root", str(root),
                "--state-dir", str(state_dir),
                "--project", "roborender-round-dagger",
                "--entity", "nvidia",
                "--expected-email", "ldu@nvidia.com",
                "--authorization-scope", "team",
                "--driver-status", str(driver),
            ]
            with mock.patch.object(
                bridge, "_load_wandb", return_value=fake
            ), mock.patch.object(
                bridge, "query_identity", return_value=identity
            ), mock.patch.object(
                bridge.campaign_state, "load_state", return_value=coordinator
            ), mock.patch.object(
                bridge,
                "_cohort_seals",
                return_value=[(0, 0, historical), (1, 0, current)],
            ), mock.patch.object(
                bridge,
                "summarize_cohort",
                side_effect=[historical_record, current_record],
            ) as summarize:
                self.assertEqual(bridge.main(arguments), 0)

            self.assertEqual(
                summarize.call_args_list,
                [
                    mock.call(historical.parent, macro_round=0, cohort_index=0),
                    mock.call(current.parent, macro_round=1, cohort_index=0),
                ],
            )
            quality_logs = [
                payload
                for payload, _step in fake.runs[0].logs
                if "rollout_quality/event_index" in payload
            ]
            self.assertEqual(
                quality_logs,
                [historical_record["metrics"], current_record["metrics"]],
            )
            cursor = json.loads(
                (state_dir / "student_quality_wandb_state.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(
                cursor["uploaded_seal_sha256_by_event"],
                {
                    "0": historical_record["seal_sha256"],
                    "100": current_record["seal_sha256"],
                },
            )

    def test_quality_log_failure_leaves_cursor_unchanged_for_replay(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = _campaign(Path(temporary) / "fixture")
            state_dir = Path(temporary) / "quality-observer"
            driver = Path(temporary) / "driver_status.json"
            driver.write_text(
                json.dumps(
                    {
                        "schema": "fast_roborender.round_dagger.production_driver_status.v1",
                        "campaign_root": str(root),
                        "state": "LAUNCHING",
                        "updated_unix_ns": 1,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            seal = Path(temporary) / "seal.json"
            seal.write_text("seal\n", encoding="utf-8")
            record = {
                "event": 0,
                "seal_sha256": bridge.sha256_file(seal),
                "metrics": {"rollout_quality/event_index": 0},
            }
            run = _FakeRun(fail_quality_log=True)
            fake = _FakeWandb(run)
            identity = SimpleNamespace(entity="nvidia", email="ldu@nvidia.com")
            arguments = [
                "--campaign-root", str(root),
                "--state-dir", str(state_dir),
                "--project", "roborender-round-dagger",
                "--entity", "nvidia",
                "--expected-email", "ldu@nvidia.com",
                "--authorization-scope", "team",
                "--driver-status", str(driver),
            ]
            with mock.patch.object(
                bridge, "_load_wandb", return_value=fake
            ), mock.patch.object(
                bridge, "query_identity", return_value=identity
            ), mock.patch.object(
                bridge, "_cohort_seals", return_value=[(0, 0, seal)]
            ), mock.patch.object(
                bridge, "summarize_cohort", return_value=record
            ):
                self.assertEqual(bridge.main(arguments), 2)

            self.assertEqual(run.finished, [2])
            cursor = json.loads(
                (state_dir / "student_quality_wandb_state.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(cursor["uploaded_seal_sha256_by_event"], {})
            # A fresh process sees no durable acknowledgement and therefore
            # replays the event instead of silently losing it.
            replay_run = _FakeRun()
            replay = _FakeWandb(replay_run)
            with mock.patch.object(
                bridge, "_load_wandb", return_value=replay
            ), mock.patch.object(
                bridge, "query_identity", return_value=identity
            ), mock.patch.object(
                bridge, "_cohort_seals", return_value=[(0, 0, seal)]
            ), mock.patch.object(
                bridge, "summarize_cohort", return_value=record
            ):
                self.assertEqual(bridge.main(arguments), 0)
            self.assertEqual(
                [
                    payload
                    for payload, _step in replay_run.logs
                    if "rollout_quality/event_index" in payload
                ],
                [record["metrics"]],
            )
            self.assertTrue(
                (state_dir / "student_quality_wandb_state.json").is_file()
            )

    def test_finish_failure_leaves_cursor_unchanged_for_replay(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = _campaign(Path(temporary) / "fixture")
            state_dir = Path(temporary) / "quality-observer"
            driver = Path(temporary) / "driver_status.json"
            driver.write_text(
                json.dumps(
                    {
                        "schema": "fast_roborender.round_dagger.production_driver_status.v1",
                        "campaign_root": str(root),
                        "state": "LAUNCHING",
                        "updated_unix_ns": 1,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            seal = Path(temporary) / "seal.json"
            seal.write_text("seal\n", encoding="utf-8")
            record = {
                "event": 0,
                "seal_sha256": bridge.sha256_file(seal),
                "metrics": {"rollout_quality/event_index": 0},
            }
            run = _FakeRun(fail_finish=True)
            fake = _FakeWandb(run)
            identity = SimpleNamespace(entity="nvidia", email="ldu@nvidia.com")
            arguments = [
                "--campaign-root", str(root),
                "--state-dir", str(state_dir),
                "--project", "roborender-round-dagger",
                "--entity", "nvidia",
                "--expected-email", "ldu@nvidia.com",
                "--authorization-scope", "team",
                "--driver-status", str(driver),
            ]
            with mock.patch.object(
                bridge, "_load_wandb", return_value=fake
            ), mock.patch.object(
                bridge, "query_identity", return_value=identity
            ), mock.patch.object(
                bridge, "_cohort_seals", return_value=[(0, 0, seal)]
            ), mock.patch.object(
                bridge, "summarize_cohort", return_value=record
            ):
                self.assertEqual(bridge.main(arguments), 2)

            self.assertEqual(run.finished, [0])
            cursor = json.loads(
                (state_dir / "student_quality_wandb_state.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(cursor["uploaded_seal_sha256_by_event"], {})

    def test_polling_keeps_session_cursor_in_memory_until_terminal_drain(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = _campaign(Path(temporary) / "fixture")
            state_dir = Path(temporary) / "quality-observer"
            state_path = state_dir / "student_quality_wandb_state.json"
            driver = Path(temporary) / "driver_status.json"
            driver.write_text("{}\n", encoding="utf-8")
            seal = Path(temporary) / "seal.json"
            seal.write_text("seal\n", encoding="utf-8")
            record = {
                "event": 0,
                "seal_sha256": bridge.sha256_file(seal),
                "metrics": {"rollout_quality/event_index": 0},
            }
            active = {
                "state": "LAUNCHING",
                "updated_unix_ns": 1,
                "collection_retry_count": 0,
                "collection_retry_limit": 2,
            }
            failed = {
                "state": "FAILED",
                "updated_unix_ns": 2,
                "collection_retry_count": 2,
                "collection_retry_limit": 2,
            }
            run = _FakeRun()
            fake = _FakeWandb(run)
            identity = SimpleNamespace(entity="nvidia", email="ldu@nvidia.com")
            arguments = [
                "--campaign-root", str(root),
                "--state-dir", str(state_dir),
                "--project", "roborender-round-dagger",
                "--entity", "nvidia",
                "--expected-email", "ldu@nvidia.com",
                "--authorization-scope", "team",
                "--driver-status", str(driver),
                "--poll-seconds", "0.01",
            ]

            def between_polls(_seconds):
                self.assertTrue(state_path.exists())
                cursor = json.loads(state_path.read_text(encoding="utf-8"))
                self.assertEqual(cursor["uploaded_seal_sha256_by_event"], {})

            with mock.patch.object(
                bridge, "_load_wandb", return_value=fake
            ), mock.patch.object(
                bridge, "query_identity", return_value=identity
            ), mock.patch.object(
                bridge, "_driver_status", side_effect=[active, failed]
            ), mock.patch.object(
                bridge, "_cohort_seals", return_value=[(0, 0, seal)]
            ), mock.patch.object(
                bridge, "summarize_cohort", return_value=record
            ) as summarize, mock.patch.object(
                bridge.time, "sleep", side_effect=between_polls
            ):
                self.assertEqual(bridge.main(arguments), 0)

            summarize.assert_called_once()
            self.assertEqual(run.finished, [0])
            cursor = json.loads(state_path.read_text(encoding="utf-8"))
            self.assertEqual(
                cursor["uploaded_seal_sha256_by_event"],
                {"0": record["seal_sha256"]},
            )

    def test_successful_finish_precedes_durable_cursor_commit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = _campaign(Path(temporary) / "fixture")
            state_dir = Path(temporary) / "quality-observer"
            state_path = state_dir / "student_quality_wandb_state.json"
            driver = Path(temporary) / "driver_status.json"
            driver.write_text(
                json.dumps(
                    {
                        "schema": "fast_roborender.round_dagger.production_driver_status.v1",
                        "campaign_root": str(root),
                        "state": "LAUNCHING",
                        "updated_unix_ns": 1,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            seal = Path(temporary) / "seal.json"
            seal.write_text("seal\n", encoding="utf-8")
            record = {
                "event": 0,
                "seal_sha256": bridge.sha256_file(seal),
                "metrics": {"rollout_quality/event_index": 0},
            }
            events: list[str] = []
            run = _FakeRun()
            original_finish = run.finish

            def finish(*, exit_code=0):
                self.assertTrue(state_path.exists())
                cursor = json.loads(state_path.read_text(encoding="utf-8"))
                self.assertEqual(cursor["uploaded_seal_sha256_by_event"], {})
                events.append("finish")
                original_finish(exit_code=exit_code)

            run.finish = finish  # type: ignore[method-assign]
            fake = _FakeWandb(run)
            identity = SimpleNamespace(entity="nvidia", email="ldu@nvidia.com")
            original_atomic_json = bridge._atomic_json

            def atomic_json(path, document):
                if path == state_path:
                    events.append(
                        "initial_cursor" if not state_path.exists() else "cursor"
                    )
                elif path.name == "student_quality_wandb_initialized.json":
                    events.append("initialized")
                original_atomic_json(path, document)

            arguments = [
                "--campaign-root", str(root),
                "--state-dir", str(state_dir),
                "--project", "roborender-round-dagger",
                "--entity", "nvidia",
                "--expected-email", "ldu@nvidia.com",
                "--authorization-scope", "team",
                "--driver-status", str(driver),
            ]
            with mock.patch.object(
                bridge, "_load_wandb", return_value=fake
            ), mock.patch.object(
                bridge, "query_identity", return_value=identity
            ), mock.patch.object(
                bridge, "_cohort_seals", return_value=[(0, 0, seal)]
            ), mock.patch.object(
                bridge, "summarize_cohort", return_value=record
            ), mock.patch.object(
                bridge, "_atomic_json", side_effect=atomic_json
            ):
                self.assertEqual(bridge.main(arguments), 0)

            self.assertEqual(
                events,
                ["initial_cursor", "initialized", "finish", "cursor"],
            )
            self.assertTrue(state_path.is_file())

    def test_unacknowledged_cursor_replays_then_acknowledged_run_requires_resume(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = _campaign(Path(temporary) / "fixture")
            campaign = artifacts.load_strict_json(root / "campaign.json")
            campaign_sha = bridge.canonical_json_sha256(campaign)
            run_id = bridge._run_id(campaign["campaign_id"], campaign_sha)
            state_dir = Path(temporary) / "quality-observer"
            state_dir.mkdir()
            state_path = state_dir / "student_quality_wandb_state.json"
            initialized_path = (
                state_dir / "student_quality_wandb_initialized.json"
            )
            seal = Path(temporary) / "seal.json"
            seal.write_text("seal\n", encoding="utf-8")
            seal_sha = bridge.sha256_file(seal)
            expected = bridge._initial_state(
                campaign_id=campaign["campaign_id"],
                campaign_sha256=campaign_sha,
                entity="nvidia",
                project="roborender-round-dagger",
                run_id=run_id,
            )
            # This cursor could have come from an old/deleted remote run. With
            # no initialization marker it must not suppress event zero.
            bridge._atomic_json(
                state_path,
                {
                    **expected,
                    "uploaded_seal_sha256_by_event": {"0": seal_sha},
                },
            )
            driver = Path(temporary) / "driver_status.json"
            driver.write_text(
                json.dumps(
                    {
                        "schema": (
                            "fast_roborender.round_dagger."
                            "production_driver_status.v1"
                        ),
                        "campaign_root": str(root),
                        "state": "LAUNCHING",
                        "updated_unix_ns": 1,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            record = {
                "event": 0,
                "seal_sha256": seal_sha,
                "metrics": {"rollout_quality/event_index": 0},
            }
            identity = SimpleNamespace(entity="nvidia", email="ldu@nvidia.com")
            arguments = [
                "--campaign-root", str(root),
                "--state-dir", str(state_dir),
                "--project", "roborender-round-dagger",
                "--entity", "nvidia",
                "--expected-email", "ldu@nvidia.com",
                "--authorization-scope", "team",
                "--driver-status", str(driver),
            ]

            interrupted = _FakeWandb()
            with mock.patch.object(
                bridge, "_load_wandb", return_value=interrupted
            ), mock.patch.object(
                bridge, "query_identity", return_value=identity
            ), mock.patch.object(
                bridge, "_write_initialized_marker",
                side_effect=RuntimeError("injected marker crash"),
            ):
                self.assertEqual(bridge.main(arguments), 2)
            self.assertEqual(interrupted.init_calls[0]["resume"], "allow")
            self.assertFalse(initialized_path.exists())
            self.assertEqual(
                json.loads(state_path.read_text(encoding="utf-8"))[
                    "uploaded_seal_sha256_by_event"
                ],
                {},
            )

            recovered = _FakeWandb()
            with mock.patch.object(
                bridge, "_load_wandb", return_value=recovered
            ), mock.patch.object(
                bridge, "query_identity", return_value=identity
            ), mock.patch.object(
                bridge, "_cohort_seals", return_value=[(0, 0, seal)]
            ), mock.patch.object(
                bridge, "summarize_cohort", return_value=record
            ) as summarize:
                self.assertEqual(bridge.main(arguments), 0)
                self.assertEqual(bridge.main(arguments), 0)

            self.assertEqual(
                [call["resume"] for call in recovered.init_calls],
                ["allow", "must"],
            )
            self.assertEqual(summarize.call_count, 1)
            first_quality = [
                payload
                for payload, _step in recovered.runs[0].logs
                if "rollout_quality/event_index" in payload
            ]
            second_quality = [
                payload
                for payload, _step in recovered.runs[1].logs
                if "rollout_quality/event_index" in payload
            ]
            self.assertEqual(first_quality, [record["metrics"]])
            self.assertEqual(second_quality, [])
            self.assertTrue(
                bridge._load_initialized_marker(initialized_path, expected)
            )

    def test_ready_marker_is_written_after_first_successful_poll(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = _campaign(Path(temporary) / "fixture")
            state_dir = Path(temporary) / "quality-observer"
            ready = state_dir / "bridge.ready.json"
            driver = Path(temporary) / "driver_status.json"
            driver.write_text(
                json.dumps(
                    {
                        "schema": "fast_roborender.round_dagger.production_driver_status.v1",
                        "campaign_root": str(root),
                        "state": "LAUNCHING",
                        "updated_unix_ns": 1,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            fake = _FakeWandb()
            identity = SimpleNamespace(entity="nvidia", email="ldu@nvidia.com")
            arguments = [
                "--campaign-root", str(root),
                "--state-dir", str(state_dir),
                "--project", "roborender-round-dagger",
                "--entity", "nvidia",
                "--expected-email", "ldu@nvidia.com",
                "--authorization-scope", "team",
                "--driver-status", str(driver),
                "--ready-marker", str(ready),
            ]
            with mock.patch.object(
                bridge, "_load_wandb", return_value=fake
            ), mock.patch.object(
                bridge, "query_identity", return_value=identity
            ):
                self.assertEqual(bridge.main(arguments), 0)

            document = json.loads(ready.read_text(encoding="utf-8"))
            self.assertEqual(document["schema"], bridge.READY_SCHEMA)
            self.assertEqual(
                document["campaign_id"],
                artifacts.load_strict_json(root / "campaign.json")["campaign_id"],
            )
            self.assertEqual(document["wandb_run_id"], fake.init_calls[0]["id"])
            self.assertGreater(document["process_id"], 0)


if __name__ == "__main__":
    unittest.main()
