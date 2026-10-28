"""Attest the native policy implementation actually imported by a worker.

The historical bootstrap spec contains a non-binding runtime_policy_sha256
from an older export. Do not rewrite that checkpoint/spec. These four module
pins were compared byte-for-byte with clean Raven commit94d7985 on2026-10-15;
they cover VisionActorObjPred, its parent, configured ResNet/MLP implementation
and observation normalizer. This is source identity, not a numerical GPU proof.
"""
from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path

RAVEN_COMMIT = "94d79851d9c471b1210be81c75116d9bea835a4d"
MODULES = {
    "gr00t.rl.trl.modules.vision_actor_critic_modules_obj_pred": "86acfbd0cf610aec49e3a11e7049b18d4ca83d3bf28bcb63fc3484683c1cb239",
    "gr00t.rl.trl.modules.vision_actor_critic_modules": "dfda777a937f1f8253e0eb8979045784da9731dfea51c9fbf4fea1ed39e00733",
    "gr00t.rl.agents.modules.modules": "f8d9b8a2a261ee0ca17306648bf2838d87947a310931ee30d38f74ed13c09965",
    "gr00t.rl.utils.running_mean_std": "821f8c5021cfefc86cc7e5c760622789984c2c9fdd7dd840fcce70ad70a97dcf",
}


def attest_policy_runtime(*, resolve_module=None, expected=None):
    pins = MODULES if expected is None else expected
    resolve = resolve_module or importlib.import_module
    records = []
    for name, wanted in pins.items():
        module = resolve(name)
        filename = getattr(module, "__file__", None)
        if not isinstance(filename, str) or not filename.endswith(".py"):
            raise RuntimeError("native policy module has no attestable Python source: " + name)
        path = Path(filename).resolve(strict=True)
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != wanted:
            raise RuntimeError("native policy implementation differs from pinned Raven source: " + name)
        records.append({"module": name, "path": str(path), "sha256": actual})
    result = {"schema": "roborender.raven-policy-runtime.v1", "raven_commit": RAVEN_COMMIT,
              "verified": True, "modules": records}
    print(json.dumps({"phase": "policy_runtime_source_verified", **result}), flush=True)
    return result
