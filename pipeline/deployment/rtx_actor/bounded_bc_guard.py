"""GPU-window BC profile with separate qualification/production admission."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import time

try:
    from .bounded_bc_storage import PROTOCOL, raw_window_bytes
except ImportError:
    from bounded_bc_storage import PROTOCOL, raw_window_bytes

GIB = 2**30
ACK = "QUALIFY_GPU_WINDOW_LANE_WRITER_NO_PRODUCTION_OR_REPLAY_V1"
PRODUCTION_ACK = "ADMITTED_B32_TOTAL10000_NATURAL_DRAIN_NO_REPLAY_V1"


def memory_gate(snapshot, available_bytes, *, num_envs, cap_bytes, saving=False, tail_growth_bytes=None,
                host_reserve_bytes=8 * GIB, peer_capture_bytes=0, strict_limit_events=False):
    """New allocation model, same caps/floors and zero swap/OOM rule.

    Eight GiB is a conservative allowance for one lane's conversions, NPZ and
    allocator scratch, NOT an experimentally proven upper bound. Copied tails
    are accounted separately: B32 retained tails alone can approach ten GiB.
    Existing host-buffer save projection must not admit this separate profile.
    """
    if (type(host_reserve_bytes) is not int or host_reserve_bytes not in (4 * GIB, 8 * GIB)
            or type(peer_capture_bytes) is not int or peer_capture_bytes not in (0, 2 * GIB)
            or (peer_capture_bytes and saving)
            or type(num_envs) is not int or not 1 <= num_envs <= 50
            or cap_bytes not in (40 * GIB, 43 * GIB)
            or type(available_bytes) is not int or available_bytes < 0
            or snapshot.get("sampling_error") or snapshot.get("memory_max") != cap_bytes
            or any(type(snapshot.get(k)) is not int or snapshot[k] < 0
                   for k in ("memory_current", "memory_peak", "memory_swap.current"))
            or any(type(snapshot.get("events", {}).get(k)) is not int or snapshot["events"][k] < 0
                   for k in ("max", "oom", "oom_kill"))):
        raise RuntimeError("bounded BC memory identity/counters changed")
    if saving and (type(tail_growth_bytes) is not int or tail_growth_bytes < 0):
        raise RuntimeError("bounded save requires explicit accepted-window tail growth accounting")
    extra = (8 * GIB + tail_growth_bytes if saving else
             max(2 * GIB, 3 * raw_window_bytes(num_envs) // 81))
    if (snapshot["memory_swap.current"] or any(snapshot["events"][k] for k in ("oom", "oom_kill"))
            or (strict_limit_events and snapshot['events']['max'])
            or snapshot["memory_current"] + extra + 2 * GIB > cap_bytes
            or snapshot["memory_peak"] > cap_bytes
            or available_bytes < extra + host_reserve_bytes + peer_capture_bytes):
        raise RuntimeError("bounded BC capture/save headroom, pressure, swap or OOM rejected")
    return dict(protocol=PROTOCOL, num_envs=num_envs, memory_max_bytes=cap_bytes,
        projected_additional_bytes=extra, host_available_bytes=available_bytes, admitted=True,
        minimum_host_reserve_bytes=host_reserve_bytes, peer_capture_reserve_bytes=peer_capture_bytes,
        recovered_limit_events=snapshot["events"]["max"],
        capacity_certified=False,
        tail_growth_bytes=tail_growth_bytes if saving else 0, projection_is_proven_upper_bound=False)


def gpu_gate(free_bytes, total_bytes, *, num_envs, captured_rows):
    if (type(free_bytes) is not int or type(total_bytes) is not int
            or not 0 <= free_bytes <= total_bytes or total_bytes <= 0
            or type(captured_rows) is not int or not 0 <= captured_rows <= 81):
        raise RuntimeError("bounded BC CUDA memory counters/window cursor changed")
    remaining = raw_window_bytes(num_envs) * (81 - captured_rows) // 81
    lane_stack = raw_window_bytes(1) // 7  # 81x480x832 float32 depth, B=1.
    # Reserve all remaining capture rows BEFORE accepting another physical step,
    # plus stack/allocator slack and the unchanged 8GiB/15% device floor.
    floor = max(8 * GIB, (15 * total_bytes + 99) // 100)
    if free_bytes < remaining + lane_stack + 2 * GIB + floor:
        raise RuntimeError("bounded BC full raw GPU window cannot fit with retained device headroom")
    return dict(free_bytes=free_bytes, total_bytes=total_bytes,
        remaining_raw_window_bytes=remaining, lane_stack_bytes=lane_stack,
        allocator_slack_bytes=2 * GIB, minimum_free_bytes=floor, admitted=True)


def __getattr__(name):
    if name != "BoundedContinuousBCTrainer":
        raise AttributeError(name)
    import torch
    from isaac6_bc_production import Isaac6BCProductionTrainer, load_request
    from isaac6_bc_session import memory_snapshot
    from continuous_bc_guard import validate_full_buffers
    from bounded_bc_storage import (archive_latest_capture, write_window, validate_device_buffers,
                                    tail_memory_budget, release_payload_cache)
    from pipeline.bc_collection.episode_drain import is_bounded_qualification
    from pipeline.bc_collection.production_gate import is_production
    from bc_resource_observer import wait_for_observer

    request = load_request()
    production = is_production(request)
    if (request.get("collection_storage_protocol") != PROTOCOL
            or (request.get("max_windows") not in (4, 8) and "diagnostic_stop" not in request
                and not is_bounded_qualification(request) and not production)
            or os.environ.get("ROBORENDER_CONTINUOUS_BC_ACK") != (PRODUCTION_ACK if production else ACK)
            or hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
               != os.environ.get("ROBORENDER_CONTINUOUS_BC_GUARD_SHA256")):
        raise RuntimeError("bounded BC source/qualification request/ack differs")
    cap, b = int(os.environ["ROBORENDER_CONTINUOUS_BC_MEMORY_BYTES"]), len(request["motions"])

    def available():
        rows = dict(line.split(":", 1) for line in Path("/run/host-meminfo").read_text().splitlines())
        return int(rows["MemAvailable"].split()[0]) * 1024

    def report(phase, value):
        with Path("/dagger-runtime/capacity-save-events.jsonl").open("a") as stream:
            stream.write(json.dumps(dict(phase=phase, monotonic_seconds=time.monotonic(), **value),
                                    sort_keys=True, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    class BoundedContinuousBCTrainer(Isaac6BCProductionTrainer):
        _bc_storage_protocol = PROTOCOL
        _bounded_saved_windows = 0

        def _check_host_memory(self, *, saving=False, tail_growth_bytes=None):
            wait_for_observer()
            sample, free = memory_snapshot(), available()
            try:
                return memory_gate(sample, free, num_envs=b, cap_bytes=cap, strict_limit_events=production,
                                   saving=saving, tail_growth_bytes=tail_growth_bytes)
            except Exception as exc:
                report("memory_rejected", dict(memory=sample, host_available_bytes=free,
                    saving=saving, tail_growth_bytes=tail_growth_bytes,
                    error_type=type(exc).__name__, error=str(exc)))
                raise

        def _bounded_device(self):
            device = torch.device(self.accelerator.device)
            if device.type != "cuda" or torch.cuda.device_count() != 1:
                raise RuntimeError("bounded storage requires the single admitted actor CUDA device")
            return torch.device("cuda", torch.cuda.current_device() if device.index is None else device.index)

        def policy_step(self, *args, **kwargs):
            if getattr(self, "_in_train_rollout", False):
                self._check_host_memory()
                device = self._bounded_device()
                gpu_gate(*torch.cuda.mem_get_info(device), num_envs=b, captured_rows=len(self._depth_chunk))
                before = len(self._depth_chunk)
                result = super().policy_step(*args, **kwargs)
                if len(self._depth_chunk) != before + 1:
                    raise RuntimeError("bounded capture must follow exactly one original policy step")
                archive_latest_capture(self, device)
                return result
            return super().policy_step(*args, **kwargs)

        def _write_collected_payload(self):
            self._bc_writer_lane_boundary(-1)
            write_window(self)
            # Release inactive raw-window allocations between windows so the
            # next full-window admission uses genuine free-device counters.
            torch.cuda.empty_cache()
            self._bounded_saved_windows += 1

        def _bc_writer_lane_boundary(self, lane):
            # Only after a lane helper has returned: no conversion arrays remain
            # live on its stack. Natural episode tails/raw GPU windows stay owned.
            from dagger_rtx.process_memory import compact_host_heap
            before = memory_snapshot()
            heap = compact_host_heap("bc_writer_lane_" + str(lane))
            after = memory_snapshot()
            row = dict(lane=lane, vector_steps=self._bc_ledger.vector_steps,
                saved_chunks=self._vg_saved_chunks, before=before, after=after,
                heap=heap, host_available_bytes=available(), monotonic_seconds=time.monotonic())
            with Path('/dagger-runtime/writer-memory.jsonl').open('a') as stream:
                stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + '\n')
                stream.flush()
                os.fsync(stream.fileno())
            if heap is None:
                raise RuntimeError('bounded writer requires explicit host heap compaction')
            self._check_host_memory()

        def _bc_payload_written(self, path, *, phase="written"):
            # The only caller supplies an exact newly emitted payload. Hashing
            # and provenance later read it again; release that cache separately.
            path = Path(path)
            if (path.parent != Path(self.vg_collect_dir)
                    or not path.name.startswith("chunk_" + self._vg_run_tag + "_")
                    or path.suffix != ".npz"):
                raise RuntimeError("cache release target differs from this collector")
            receipt = release_payload_cache(path)
            with Path('/dagger-runtime/output-cache-events.jsonl').open('a') as stream:
                stream.write(json.dumps(dict(phase=phase, path=str(path), **receipt)) + '\n')
                stream.flush()
                os.fsync(stream.fileno())

        def _save_collected_chunks(self):
            validate_full_buffers(self, b)
            device = self._bounded_device()
            validate_device_buffers(self, device)
            tails = tail_memory_budget(self)
            decision = self._check_host_memory(saving=True, tail_growth_bytes=tails["additional_tail_bytes"])
            graphics = gpu_gate(*torch.cuda.mem_get_info(device), num_envs=b, captured_rows=81)
            report("presave", dict(**decision, gpu=graphics, copied_tails=tails, memory=memory_snapshot(),
                vector_steps=self._bc_ledger.vector_steps,
                emitted_chunks_per_lane=list(self._bc_ledger.emitted_counts)))
            completed = False
            first = self._vg_saved_chunks
            try:
                result = super()._save_collected_chunks()
                for index in range(first, self._vg_saved_chunks):
                    self._bc_payload_written(Path(self.vg_collect_dir) /
                        f"chunk_{self._vg_run_tag}_{index:07d}.npz", phase="indexed")
                completed = True
                return result
            finally:
                after = memory_snapshot()
                stat = after.get("stat", {})
                report("postsave", dict(protocol=PROTOCOL, completed_bounded_save=completed,
                    native_payload_writer_unchanged=False, memory=after,
                    vector_steps=self._bc_ledger.vector_steps,
                    emitted_chunks_per_lane=list(self._bc_ledger.emitted_counts),
                    memory_current_bytes=after.get("memory_current"), memory_peak_bytes=after.get("memory_peak"),
                    anon_bytes=stat.get("anon"), shmem_bytes=stat.get("shmem"), kernel_bytes=stat.get("kernel"),
                    file_bytes=stat.get("file"), host_available_bytes=available()))

        def _collection_storage_receipt(self):
            import bounded_bc_storage
            return dict(protocol=PROTOCOL, completed_windows=self._bounded_saved_windows,
                writer_sha256=hashlib.sha256(Path(bounded_bc_storage.__file__).read_bytes()).hexdigest(),
                capture_device=str(self._bounded_device()), num_envs=b, frames_per_window=81,
                logical_scenes=1, cpu_conversion_lanes=1, production_admitted=False)

    globals()[name] = BoundedContinuousBCTrainer
    return BoundedContinuousBCTrainer
