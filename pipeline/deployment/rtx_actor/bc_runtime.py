"""Shared BC runtime boundary: paths and Docker assembly, never env.step.

New standalone collection supplies registry-bound canonical locations. Historical
cohort callers retain their old defaults unless they explicitly supply paths.
This module neither starts workloads nor grants source/capacity admission.
"""
from dataclasses import dataclass
from pathlib import Path
import re
import sys

from roborender.infrastructure.paths import load_registry
from pipeline.deployment.rtx_actor.isaac6_bc_production import (
    validate_collection_request, native_overrides,
)

CANONICAL_ROOT = Path('/data1/workspaces/roborender')
PATH_KEYS = ('run_root', 'bc_collection_runs', 'isaac_cache', 'bc_runtime_bundle', 'render_anchors')


@dataclass(frozen=True)
class RuntimePaths:
    registry: object

    def get(self, key):
        return self.registry.resolve('leo', key)

    def mounts(self, physical_gpu):
        if physical_gpu not in (0, 1):
            raise ValueError('Leo collection owns only GPU0/GPU1')
        return dict(raven=self.get('bc_runtime_bundle'), anchors=self.get('render_anchors'),
                    cache=self.get('isaac_cache') / f'leo-gpu{physical_gpu}')

    def validate_run(self, runtime):
        runtime = Path(runtime)
        root = self.get('bc_collection_runs')
        if (runtime.parent != root or runtime.resolve(strict=False) != runtime
                or runtime.exists() or not re.fullmatch('[a-z0-9][a-z0-9-]{0,80}', runtime.name)):
            raise ValueError('new physical scoped BC runtime required; no resume/replay path')
        return runtime

    def check_assets(self, physical_gpu):
        for key, path in self.mounts(physical_gpu).items():
            if path.resolve(strict=False) != path:
                raise ValueError('runtime mounts must use physical canonical paths: ' + key)
            if key != 'cache' and not path.is_dir():
                raise ValueError('required runtime asset directory missing: ' + key)
        root = self.get('run_root')
        if root.resolve(strict=True) != root:
            raise ValueError('canonical lock root changed')

    def freeze(self, runtime):
        return self.registry.freeze('leo', Path(runtime) / 'storage-paths.json', keys=PATH_KEYS)


def load_paths(source):
    registry = load_registry(Path(source) / 'configs/storage_paths.json')
    paths = RuntimePaths(registry)
    # Location configuration is not authorization to widen the execution jail.
    required = dict(run_root=CANONICAL_ROOT / 'runs',
                    bc_collection_runs=CANONICAL_ROOT / 'runs/bc-continuous',
                    isaac_cache=CANONICAL_ROOT / 'caches/isaac')
    for key, value in required.items():
        if paths.get(key) != value:
            raise ValueError('canonical runtime root differs: ' + key)
    for key in ('bc_runtime_bundle', 'render_anchors'):
        path = paths.get(key)
        root = CANONICAL_ROOT / 'simulation_assets'
        if not path.is_relative_to(root) or path == root:
            raise ValueError('canonical simulation asset root required: ' + key)
    return paths


def build_command(*, request, request_path, request_sha256, name, runtime, overlay,
                  source, physical_gpu, memory_gib=32, base_builder=None, image_id=None, runtime_paths=None):
    request = validate_collection_request(request)
    if (request["output_dir"] != "/dagger-runtime/bc_isaac"
            or physical_gpu not in (0, 1) or not 21 <= memory_gib <= 40
            or not re.fullmatch(r"bc-cohort-leo-[a-z0-9-]{1,80}", name)
            or not re.fullmatch(r"[0-9a-f]{64}", request_sha256)):
        raise ValueError("invalid scoped native BC launch binding")
    if base_builder is None:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from qualify_native_isaac6 import command, IMAGE
        base_builder, image_id = command, IMAGE
        if runtime_paths is not None:
            from functools import partial
            base_builder = partial(command, storage_paths=runtime_paths.mounts(physical_gpu))
    if not isinstance(image_id, str) or not re.fullmatch(r"sha256:[0-9a-f]{64}", image_id):
        raise ValueError("an immutable native image ID is required")
    cmd = base_builder(name, runtime, overlay, Path(source) / "pipeline/dagger/rtx/code",
        len(request["motions"]), mode="bc-collect", physical_gpu=physical_gpu,
        seed=request["process_seed"], memory_gib=memory_gib, compact_host_heap=True,
        deployment_host="leo-rtx-6k")
    overrides = native_overrides(request)
    changed = set()
    for index, value in enumerate(cmd):
        key = value.split("=", 1)[0]
        if key in overrides:
            cmd[index] = key + "=" + overrides[key]
            changed.add(key)
        if value.startswith("PYTHONPATH="):
            cmd[index] = value.replace("PYTHONPATH=", "PYTHONPATH=/bc-production-source:", 1)
    cmd += [key + "=" + value for key, value in overrides.items() if key not in changed]
    extra = ["--mount", f"type=bind,src={Path(source) / 'pipeline/deployment/rtx_actor'},dst=/bc-production-source,readonly",
             "--mount", f"type=bind,src={request_path},dst=/run/bc-request.json,readonly",
             "--env", "ROBORENDER_BC_COLLECTION_REQUEST=/run/bc-request.json",
             "--env", f"ROBORENDER_BC_COLLECTION_REQUEST_SHA256={request_sha256}"]
    cmd[cmd.index(image_id):cmd.index(image_id)] = extra
    return cmd
