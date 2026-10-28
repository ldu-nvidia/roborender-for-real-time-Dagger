from test.deployment.source_paths import PIPELINE_ROOT
import json
import hashlib
import time
import subprocess
from pathlib import Path

import pytest

from qualify_native_isaac6 import command, GPU_UUID


def plan(tmp_path, **kwargs):
    (tmp_path / "port-manifest.json").write_text(json.dumps({"files": []}))
    return command("test-container", tmp_path / "out", tmp_path,
                   tmp_path / "code", 2, **kwargs)


def test_capture_has_no_host_network_or_remote_credentials(tmp_path):
    cmd = plan(tmp_path)
    assert cmd[cmd.index("--network") + 1] == "bridge"
    assert cmd[cmd.index("--gpus") + 1] == "device=" + GPU_UUID
    assert not any("TOKEN_FILE" in value for value in cmd)
    assert "faulthandler.dump_traceback_later" not in " ".join(cmd)


def test_parallel_panel_arms_use_only_dedicated_gpus_and_noswap_heap_profile(tmp_path):
    from run_paired_student_success import build_command
    from qualify_native_isaac6 import BETA_BENCHMARK_GPUS
    (tmp_path / "port-manifest.json").write_text(json.dumps({"files": []}))
    for gpu in (1, 2):
        cmd = build_command("panel-test", tmp_path / "out", tmp_path,
                            tmp_path / "code", tmp_path / "source", tmp_path / "checkpoint.pt",
                            "a" * 64, "bc" if gpu == 1 else "dagger", 1103, [],
                            physical_gpu=gpu, memory_gib=21, compact_host_heap=True)
        assert cmd[cmd.index("--gpus") + 1] == "device=" + BETA_BENCHMARK_GPUS[gpu]
        assert cmd[cmd.index("--memory") + 1] == cmd[cmd.index("--memory-swap") + 1] == "21g"
        assert "ROBORENDER_RTX_COMPACT_HEAP=1" in cmd
        assert "algo.config.num_learning_epochs=0" in cmd
        assert "domain_rand.image_augmentation.enabled=false" in cmd


def test_warm_panel_command_mounts_two_pinned_policies_in_one_isaac_lifetime(tmp_path):
    from run_small_batch_success import build_warm_command
    from leo_binding import GPU_UUIDS
    (tmp_path / "port-manifest.json").write_text(json.dumps({"files": []}))
    request = tmp_path / "request.json"
    request.write_text(json.dumps({"panel_sha256": "a" * 64, "cohort_limit": 2}))
    baseline, candidate = tmp_path / "baseline.pt", tmp_path / "candidate.pt"
    cmd = build_warm_command("warm-test", tmp_path / "out", tmp_path, tmp_path / "source",
        tmp_path / "panel.json", request, "b" * 64,
        {"bc": (baseline, "c" * 64), "dagger": (candidate, "d" * 64)}, [], 0, "leo-rtx-6k")
    assert cmd[cmd.index("--gpus") + 1] == "device=" + GPU_UUIDS[0]
    assert cmd[cmd.index("--memory") + 1] == cmd[cmd.index("--memory-swap") + 1] == "40g"
    assert f"type=bind,src={baseline},dst=/diagnostic-assets/best_validation.pt,readonly" in cmd
    assert f"type=bind,src={candidate},dst=/diagnostic-assets/warm-candidate.pt,readonly" in cmd
    assert "trainer._target_=isaac6_panel_evaluator.WarmPanelEvaluator" in cmd
    assert "algo.config.num_learning_epochs=0" in cmd
    assert "domain_rand.image_augmentation.enabled=false" in cmd
    assert "WARM_PANEL_REQUEST_SHA256=" + "b" * 64 in cmd
    assert "--no-healthcheck" in cmd  # External runner owns evaluation health.
    assert "--health-cmd" not in cmd and "--health-start-period" not in cmd


def test_global_swap_change_does_not_kill_nonswapping_actors_with_ram_headroom():
    from run_throughput_pair import memory_safety
    initial = {"SwapFree": 32381560}
    current = {"SwapFree": 32243120, "MemAvailable": 26217620}
    samples = [{"pid": 101, "memory_swap.current": 0, "events": {"oom_kill": 0}},
               {"pid": 102, "memory_swap.current": 0, "events": {"oom_kill": 0}}]
    report = memory_safety(initial, current, samples)
    assert report["host_swap_growth_kib"] == 138440
    assert report["host_swap_outside_owned_cgroups"] and report["owned_swap_bytes"] == 0
    with pytest.raises(RuntimeError, match="attribution"):
        memory_safety(initial, current, [{"pid": 101, "sampling_error": "unavailable"}])
    with pytest.raises(RuntimeError, match="forbidden swap"):
        memory_safety(initial, current, [{"pid": 101, "memory_swap.current": 4096}])
    with pytest.raises(RuntimeError, match="OOM"):
        memory_safety(initial, current, [{"pid": 101, "oom_killed": True}])
    with pytest.raises(RuntimeError, match="RAM safety"):
        memory_safety(initial, {**current, "MemAvailable": 5 * 1024**2}, samples)


class ExitingActor:
    def __init__(self, code=None, wait_code=None):
        self.code, self.wait_code, self.waits = code, wait_code, []

    def poll(self):
        return self.code

    def wait(self, timeout):
        self.waits.append(timeout)
        if self.wait_code is None:
            raise subprocess.TimeoutExpired("owned-native-actor", timeout)
        self.code = self.wait_code
        return self.code


def test_removed_cgroup_after_native_pass_is_terminal_not_a_swap_failure(tmp_path, monkeypatch):
    import run_throughput_pair as runner
    receipt = tmp_path / "native-service-result.json"
    receipt.write_text(json.dumps({"result": "PASS"}))
    monkeypatch.setattr(runner, "container_memory_sample", lambda n: pytest.fail("already exited"))
    terminal = runner.owned_memory_sample("owned-actor", ExitingActor(0), receipt)
    assert terminal["lifecycle"] == "completed_attested"
    assert terminal["completion_receipt_sha256"] == hashlib.sha256(receipt.read_bytes()).hexdigest()
    report = runner.memory_safety({"SwapFree": 32000000},
        {"SwapFree": 31388440, "MemAvailable": 40208688},
        [terminal, {"pid": 102, "memory_swap.current": 0}])
    assert report["owned_swap_bytes"] == 0


def test_exit_during_cgroup_read_requires_both_zero_exit_and_pass(tmp_path, monkeypatch):
    import run_throughput_pair as runner
    receipt = tmp_path / "native-service-result.json"
    receipt.write_text('{"result":"PASS"}')
    process = ExitingActor(wait_code=0)
    monkeypatch.setattr(runner, "container_memory_sample", lambda n: {"sampling_error": "removed"})
    terminal = runner.owned_memory_sample("owned-actor", process, receipt)
    assert terminal["exit_code"] == 0 and process.waits == [1]


@pytest.mark.parametrize("code,body", [(0, None), (0, '{"result":"FAIL"}'), (1, '{"result":"PASS"}')])
def test_unsuccessful_or_unattested_native_exit_cannot_bypass_guard(tmp_path, monkeypatch, code, body):
    import run_throughput_pair as runner
    receipt = tmp_path / "native-service-result.json"
    if body is not None:
        receipt.write_text(body)
    with pytest.raises(RuntimeError, match="native actor"):
        runner.owned_memory_sample("owned-actor", ExitingActor(code), receipt)


def test_pass_file_before_native_exit_is_not_completion(tmp_path, monkeypatch):
    import run_throughput_pair as runner
    receipt = tmp_path / "native-service-result.json"
    receipt.write_text('{"result":"PASS"}')
    process = ExitingActor()
    monkeypatch.setattr(runner, "container_memory_sample", lambda n: {"sampling_error": "missing"})
    sample = runner.owned_memory_sample("owned-actor", process, receipt)
    assert process.waits == [1] and "lifecycle" not in sample
    with pytest.raises(RuntimeError, match="attribution"):
        runner.memory_safety({"SwapFree": 32000000},
            {"SwapFree": 31388440, "MemAvailable": 40208688}, [sample])


def test_unattested_stopped_marker_does_not_excuse_missing_memory_counters():
    from run_throughput_pair import memory_safety
    for sample in [{"pid": 0}, {"lifecycle": "completed_attested"},
                   {"lifecycle": "completed_attested", "exit_code": 1,
                    "native_pass_receipt": True, "completion_receipt_sha256": "a" * 64}]:
        with pytest.raises(RuntimeError, match="attribution"):
            memory_safety({"SwapFree": 32000000},
                {"SwapFree": 31388440, "MemAvailable": 40208688}, [sample])


def test_warm_evaluator_exit_uses_its_own_complete_receipt(tmp_path):
    from run_throughput_pair import owned_memory_sample
    receipt = tmp_path / "warm-session.json"
    receipt.write_text('{"complete":true,"optimizer_steps":0}')
    sample = owned_memory_sample("owned-evaluator", ExitingActor(0), receipt,
        completion_field="complete", completion_value=True)
    assert sample["native_pass_receipt"] is True
    receipt.write_text('{"complete":1}')
    with pytest.raises(RuntimeError, match="PASS receipt"):
        owned_memory_sample("owned-evaluator", ExitingActor(0), receipt,
            completion_field="complete", completion_value=True)


def test_only_terminal_native_attestation_is_host_readable(tmp_path):
    import ast
    import stat
    from dagger_rtx.durable import atomic_write_json
    source = PIPELINE_ROOT / "dagger/rtx/code/src/dagger_rtx/native_service_trainer.py"
    calls = [n for n in ast.walk(ast.parse(source.read_text())) if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Name) and n.func.id == "atomic_write_json"
        and n.args and isinstance(n.args[0], ast.BinOp)
        and isinstance(n.args[0].right, ast.Constant)
        and n.args[0].right.value == "native-service-result.json"]
    assert len(calls) == 1
    assert {k.arg: ast.literal_eval(k.value) for k in calls[0].keywords} == {"mode": 0o644}
    atomic_write_json(tmp_path / "terminal.json", {"result": "PASS"}, mode=0o644)
    atomic_write_json(tmp_path / "private.json", {"example": "private"})
    assert stat.S_IMODE((tmp_path / "terminal.json").stat().st_mode) == 0o644
    assert stat.S_IMODE((tmp_path / "private.json").stat().st_mode) == 0o600


def test_pair_capture_waits_for_both_native_owners_without_network(tmp_path):
    cmd = plan(tmp_path, mode="capture-only", run_id="rtx-beta-migration-pair-capture",
               benchmark_barrier=tmp_path / "barrier", physical_gpu=2)
    assert "DAGGER_BENCHMARK_BARRIER_DIR=/benchmark-barrier" in cmd
    assert "DAGGER_ACTOR_PREFIX=rtx-beta-migration-pair-capture-g2" in cmd
    assert cmd[cmd.index("--network") + 1] == "bridge"
    assert not any("TOKEN_FILE" in value for value in cmd)


@pytest.mark.parametrize("mode", ["isolated-one-step", "single-window"])
def test_crosshost_is_bounded_single_gateway_native_code(tmp_path, mode):
    cmd = plan(tmp_path, mode=mode, run_id="rtx-beta-migration-test",
               token_file=Path("/private/scoped.token"), h200_pid=123)
    assert cmd[cmd.index("--network") + 1] == "host"
    assert "DAGGER_H200_RPC_WORKERS=1" in cmd
    assert "DAGGER_EXPECTED_H200_PIDS=123" in cmd
    assert "DAGGER_PHYSICAL_GPU=1" in cmd
    assert "trainer._target_=dagger_rtx.native_service_trainer.NativeDaggerRTXServiceTrainer" in cmd
    assert "type=bind,src=/private/scoped.token,dst=/run/roborender-h200.token,readonly" in cmd
    assert "--privileged" not in cmd
    assert cmd[cmd.index("--health-cmd") + 1] == "python -m dagger_rtx.container_health"
    assert cmd[cmd.index("--health-start-period") + 1] == "600s"
    assert "--restart" not in cmd


def test_cannot_start_production_or_unattested_gateway(tmp_path):
    with pytest.raises(ValueError):
        plan(tmp_path, mode="production")
    with pytest.raises(ValueError):
        plan(tmp_path, mode="isolated-one-step")


def test_bc_collect_is_original_teacher_only_mode_without_h200(tmp_path):
    cmd = plan(tmp_path, mode="bc-collect")
    assert cmd[cmd.index("--network") + 1] == "bridge"
    assert not any("TOKEN_FILE" in value for value in cmd)
    assert "+exp=loco_manip/wsdpt_videogen_collect" in cmd
    assert "algo.config.videogen.mode=collect" in cmd
    assert "algo.config.videogen.rollout_action_source=teacher" in cmd
    assert "algo.trl.num_total_batches=3" in cmd
    assert "algo.config.num_learning_epochs=0" in cmd
    assert not any("raytraced_mix_prob=" in value for value in cmd)


def test_pair_capacity_binds_gpu2_and_four_gateways(tmp_path):
    cmd = plan(tmp_path, mode="capacity-probe", run_id="rtx-beta-migration-sweep",
               token_file=Path("/private/scoped.token"), physical_gpu=2,
               gateway_pids=[100, 101, 102, 103], memory_gib=24,
               benchmark_barrier=tmp_path / "barrier", seed=7)
    assert "DAGGER_PHYSICAL_GPU=2" in cmd
    assert "DAGGER_H200_RPC_WORKERS=4" in cmd
    assert "DAGGER_CAPACITY_STEPS=10" in cmd
    assert "DAGGER_NATIVE_STEP_ACK=I_ACKNOWLEDGE_TEN_NATIVE_VECTOR_STEPS" in cmd
    assert "seed=7" in cmd
    assert cmd[cmd.index("--memory-swap") + 1] == "24g"
    assert "DAGGER_BENCHMARK_BARRIER_DIR=/benchmark-barrier" in cmd


def test_benchmark_cannot_use_gpu0_or_unbounded_memory(tmp_path):
    with pytest.raises(ValueError):
        plan(tmp_path, physical_gpu=0)
    with pytest.raises(ValueError):
        plan(tmp_path, memory_gib=100)


def test_allocator_profile_does_not_change_algorithm_or_ram_cap(tmp_path):
    old = plan(tmp_path, memory_gib=22)
    new = plan(tmp_path, memory_gib=22, compact_host_heap=True)
    extras = {"ROBORENDER_RTX_COMPACT_HEAP=1", "MALLOC_ARENA_MAX=2", "MALLOC_TRIM_THRESHOLD_=131072"}
    assert extras.issubset(new)
    stripped = []
    index = 0
    while index < len(new):
        if new[index] == "--env" and new[index + 1] in extras:
            index += 2
        else:
            stripped.append(new[index])
            index += 1
    assert stripped == old


@pytest.mark.parametrize("num_envs", [4, 6, 8])
def test_three_window_plan_needs_explicit_config_and_budget(tmp_path, num_envs):
    frozen = tmp_path / "frozen.json"
    frozen.write_text("{}")
    (tmp_path / "port-manifest.json").write_text(json.dumps({"files": []}))
    ssh_config, known_hosts = tmp_path / "ssh.config", tmp_path / "known_hosts"
    ssh_config.touch(mode=0o600)
    known_hosts.touch(mode=0o600)
    supervisor = tmp_path / "supervisor.json"
    forwards = [{"local_port": 18610, "remote_port": 18610,
                 "expected_ready": {"ok": True, "role": "coordinator", "pid": 10}}]
    forwards += [{"local_port": 18620 + i, "remote_port": 18620 + i,
                  "expected_ready": {"ok": True, "role": "render_student", "pid": 11 + i}}
                 for i in range(1 if num_envs == 8 else 4)]
    supervisor.write_text(json.dumps({"schema": "roborender.ssh-supervisor.v1",
        "run_id": "rtx-beta-migration-runtime-test", "ssh_config": str(ssh_config),
        "known_hosts": str(known_hosts), "ssh_host": "scoped-host", "lifetime_seconds": 1800,
        "forwards": forwards}))
    status = tmp_path / "supervisor-status"
    status.mkdir()
    digest = hashlib.sha256(supervisor.read_bytes()).hexdigest()
    (status / "status.json").write_text(json.dumps({"schema": "roborender.ssh-supervisor.v1",
        "run_id": "rtx-beta-migration-runtime-test", "config_sha256": digest,
        "state": "ready", "updated_at_ns": time.time_ns(),
        "forwards": [{**row, "healthy": True} for row in forwards]}))
    cmd = command("runtime-test", tmp_path / "out", tmp_path, tmp_path / "code", num_envs,
                  mode="multi-window-check", run_id="rtx-beta-migration-runtime-test",
                  token_file=Path("/private/token"), physical_gpu=1 if num_envs == 8 else 2,
                  gateway_pids=[11] if num_envs == 8 else [11, 12, 13, 14],
                  qualification_config=frozen, supervisor_config=supervisor,
                  supervisor_status_dir=status)
    assert "DAGGER_NATIVE_STEP_ACK=I_ACKNOWLEDGE_THREE_WINDOWS_AND_REAL_LEARNER_UPDATES" in cmd
    assert "DAGGER_QUALIFICATION_CONFIG=/run/runtime-check.json" in cmd
    assert ("DAGGER_PHYSICAL_GPU=1" if num_envs == 8 else "DAGGER_PHYSICAL_GPU=2") in cmd
    assert "DAGGER_CONNECTION_CONFIG_SHA256=" + digest in cmd
    assert "DAGGER_CONNECTION_STATUS=/run/roborender-connection/status.json" in cmd
    assert f"type=bind,src={status},dst=/run/roborender-connection,readonly" in cmd
    assert not any("src=" + str(status / "status.json") in arg for arg in cmd)
    with pytest.raises(ValueError, match="supervised"):
        command("runtime-test", tmp_path / "out", tmp_path, tmp_path / "code", 6,
                mode="multi-window-check", run_id="rtx-beta-migration-runtime-test",
                token_file=Path("/private/token"), gateway_pids=[11, 12, 13, 14],
                qualification_config=frozen)


def test_single_actor_ram_budget_cannot_expand_the_pair():
    from run_throughput_pair import validate_layout
    assert validate_layout([1], 40, 1) == 46
    assert validate_layout([1, 2], 21, 4) == 48
    assert validate_layout([1, 2], 23, 4) == 52
    for gpus, memory, fanout in [([0], 40, 1), ([2], 40, 1), ([1, 2], 40, 4),
                                  ([1], 80, 1), ([1], 40, 4)]:
        with pytest.raises(ValueError):
            validate_layout(gpus, memory, fanout)
