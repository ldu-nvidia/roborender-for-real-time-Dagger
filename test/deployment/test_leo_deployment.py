"""Leo host portability without changing rollout/learning/evaluation science."""
from test.deployment.source_paths import DEPLOYMENT_ROOT, PIPELINE_ROOT
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import pytest

from pipeline.dagger.beta_campaign import make_first_round_config, next_round_config
from pipeline.dagger.rtx.code.src.dagger_rtx.beta_round import (
    validate_config, canonical, driver_config, gpu_uuids, budgets, acknowledgement, deployment_host)
from pipeline.deployment.beta_site import Site, validate_site
from pipeline.deployment.h200.beta_pod import name_for


def config(tmp_path, smoke=True):
    return make_first_round_config("dagger-leo-" + ("smoke-" if smoke else "") + "host-test", seed=610131,
        checkpoint=tmp_path / "policy.pt", checkpoint_sha="a"*64, panel=tmp_path / "panel.json",
        panel_sha="b"*64, source_sha="c"*64, gateway_pids=[101, 102, 103, 104],
        smoke=smoke, deployment_host="leo-rtx-6k")


@pytest.mark.parametrize("smoke", [True, False])
def test_host_bound_round_lineage_and_unchanged_budgets(tmp_path, smoke):
    c = config(tmp_path, smoke)
    digest = hashlib.sha256(canonical(c)).hexdigest()
    for rid, beta in enumerate((.75, .5, .25, 0.), 1):
        assert validate_config(c, runtime=True) is c
        assert deployment_host(c) == "leo-rtx-6k"
        assert list(gpu_uuids(c)) == [0, 1] and c["forbidden_gpu_indices"] == [2]
        assert driver_config(c, digest)["physical_rtx_gpus"] == [0, 1]
        assert budgets(c)[:2] == ((3, 10) if smoke else (50, 256))
        assert c["round_id"] == rid and c["teacher_beta"] == beta
        assert "LEO" in acknowledgement(c)
        if rid < 4:
            c = next_round_config(c, digest, "d"*64, "/next/policy.pt", [111, 112, 113, 114])


@pytest.mark.parametrize("field", ["schema", "run_id", "uuid", "forbidden"])
def test_reject_mixed_beta_leo_identity(tmp_path, field):
    c = config(tmp_path)
    if field == "schema": c["schema"] = "roborender.cross_gpu_dagger_smoke.beta.v1"
    if field == "run_id": c["run_id"] = c["run_id"].replace("leo", "beta")
    if field == "uuid": c["rtx_actors"][0]["gpu_uuid"] = "GPU-wrong"
    if field == "forbidden": c["forbidden_gpu_indices"] = [0]
    with pytest.raises(ValueError): validate_config(c)


def test_native_command_uses_leo_actor_rank_seed_and_exact_uuid(tmp_path):
    from qualify_native_isaac6 import command
    c = config(tmp_path)
    path = tmp_path / "round.json"
    path.write_bytes(canonical(c))
    (tmp_path / "port-manifest.json").write_text('{"files":[]}')
    ssh, known = tmp_path / "ssh.conf", tmp_path / "known_hosts"
    ssh.touch(mode=0o600)
    known.touch(mode=0o600)
    forwards = [{"local_port": 18610, "remote_port": 18610,
        "expected_ready": {"ok": True, "role": "coordinator", "pid": 99}}]
    forwards += [{"local_port": 18620+i, "remote_port": 18620+i,
        "expected_ready": {"ok": True, "role": "render_student", "pid": 101+i}} for i in range(4)]
    supervisor = tmp_path / "supervisor.json"
    supervisor.write_bytes(canonical({"schema": "roborender.ssh-supervisor.v1", "run_id": c["run_id"],
        "ssh_config": str(ssh), "known_hosts": str(known), "ssh_host": "leo-pinned-test",
        "lifetime_seconds": 1800, "forwards": forwards}))
    status = tmp_path / "supervisor"
    status.mkdir()
    (status / "status.json").write_bytes(canonical({"schema": "roborender.ssh-supervisor.v1", "run_id": c["run_id"],
        "config_sha256": hashlib.sha256(supervisor.read_bytes()).hexdigest(), "state": "ready",
        "updated_at_ns": time.time_ns(), "forwards": [{**r, "healthy": True} for r in forwards]}))
    code = PIPELINE_ROOT / "dagger/rtx/code"
    for rank, (gpu, uuid) in enumerate(gpu_uuids(c).items()):
        cmd = command("leo-test", tmp_path / "out", tmp_path, code, 6,
            mode="beta-production", run_id=c["run_id"], token_file=tmp_path / "token",
            physical_gpu=gpu, seed=c["seed"]+rank, gateway_pids=[101,102,103,104], memory_gib=21,
            qualification_config=path, compact_host_heap=True, deployment_host="leo-rtx-6k",
            supervisor_config=tmp_path / "supervisor.json", supervisor_status_dir=tmp_path / "supervisor")
        assert cmd[cmd.index("--gpus")+1] == "device=" + uuid
        assert f"seed={c['seed']+rank}" in cmd
        assert "algo.config.init_at_random_ep_len=true" in cmd
        assert "++algo.config.raven_exact_raytraced_augmentation=true" in cmd
        with pytest.raises(ValueError):
            command("wrong-host", tmp_path / "out", tmp_path, code, 6,
                mode="beta-production", run_id=c["run_id"], token_file=tmp_path / "token",
                physical_gpu=1, seed=c["seed"], gateway_pids=[101,102,103,104], memory_gib=21,
                qualification_config=path, compact_host_heap=True)


def test_site_and_allocator_are_leo_owned_not_beta(tmp_path):
    site = json.loads((DEPLOYMENT_ROOT / "profiles/beta_site.direct.example.json").read_text())
    site.update(schema="roborender.leo-site.v1", rtx_ssh_host="roborender-leo-direct",
        broker_ssh_host="roborender-leo-direct", ssh_config="/workspace/.ssh/roborender_leo_access_config")
    assert validate_site(site) == site
    s = Site.__new__(Site)
    s.site, s.pod = site, name_for("dagger-leo-smoke-test", 1)
    assert s.pod.startswith("leo-rr-leo-")
    assert s.gpu_proxy("beta") == [site["broker_python"], site["broker_proxy"], s.pod]
    assert "roborender-leo-direct" in s.gpu_proxy("local")
    with pytest.raises(ValueError): validate_site(dict(site, rtx_ssh_host="roborender-beta-direct"))


def test_access_binding_reuses_supervisor_without_changing_beta():
    root = DEPLOYMENT_ROOT / "connectivity"
    code = """import leo_access,leo_cpu_endpoint
b=leo_access.bind();c=leo_cpu_endpoint.bind()
assert b.PORT==c.PORT==22033 and b.BETA_HOST==c.BETA_HOST=='leo-rtx-6k'
assert b.UNIT_NAME=='roborender-leo-access.service'
unit=b.service_unit(b.SDK_PYTHON,b.BASE/'source/beta_access.py',b.BASE/'config.json')
assert 'source/leo_access.py' in unit and 'Restart=always' in unit
assert 'roborender-beta-access' not in unit
line=c.authorized_line('ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIM42zyxQDsdXRVv/+8g2idOnIwczinJB7ebSuMs/1Q4x')
assert '127.0.0.1:22033' in line and 'leo_cpu_endpoint.py health' in line
assert '22032' not in line
"""
    subprocess.run([sys.executable, "-c", code], cwd=root, check=True)


def test_leo_supervisor_upgrade_accepts_only_exact_owned_unit():
    root = DEPLOYMENT_ROOT / "connectivity"
    code = """import leo_access,hashlib
from pathlib import Path
b=leo_access.bind();config=b.BASE/'config.json'
script=Path('/home/ldu/roborender/runtime/connectivity-releases/leo-access-20261013T2002/leo_access.py')
old=b.service_unit(b.SDK_PYTHON,script,config).replace('WatchdogSec=180\\nNotifyAccess=main\\n','')
sha=lambda text:hashlib.sha256(text.encode()).hexdigest()
leo_access.checked_old_unit(b,old,sha(old),config)
new=b.service_unit(b.SDK_PYTHON,script,config)
leo_access.checked_old_unit(b,new,sha(new),config)
for bad,digest in [(old,'0'*64),(old+'Environment=UNREVIEWED=1\\n',sha(old+'Environment=UNREVIEWED=1\\n')),
                   (old.replace('leo_access.py','beta_access.py'),sha(old.replace('leo_access.py','beta_access.py'))),
                   (old.replace('/runtime/connectivity-releases/','/unowned/'),sha(old.replace('/runtime/connectivity-releases/','/unowned/')))]:
    try:leo_access.checked_old_unit(b,bad,digest,config)
    except RuntimeError:pass
    else:raise AssertionError('unsafe replacement admitted')
"""
    subprocess.run([sys.executable, "-c", code], cwd=root, check=True)
