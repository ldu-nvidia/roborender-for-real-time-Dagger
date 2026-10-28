"""Opt-in native actor barrier for MSE-selected rounds; no simulator resets."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from .beta_round import canonical
from .contracts import require_sha256
from .qualification_driver import BetaRoundDriver, actor_ids


SCHEMA = "roborender.dagger-mse-round-protocol.v1"
FIXED100_SCHEMA = "roborender.dagger-fixed100-round-protocol.v1"


def read_mse_protocol(path: Path, raw_sha256: str, config: dict) -> dict:
    """CPU preflight; the sidecar's H200 paths need not exist on the RTX host."""
    require_sha256(raw_sha256, "MSE sidecar file hash")
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != raw_sha256:
        raise ValueError("MSE sidecar file digest changed")
    value = json.loads(raw)
    fields = {"schema", "run_id", "round_id", "initial_checkpoint_sha256",
              "validation_manifest_path", "validation_manifest_sha256", "interval_iterations",
              "max_iterations", "min_validation_epochs", "patience", "relative_improvement",
              "absolute_improvement", "final_test"}
    from .beta_round import is_fixed100
    fixed = is_fixed100(config)
    if fixed:
        fields = {"schema", "run_id", "round_id", "initial_checkpoint_sha256",
                  "interval_iterations", "max_iterations", "final_test",
                  "full_batch_environments", "rollout_steps", "checkpoint_selection"}
    if (set(value) != fields or value["schema"] != (FIXED100_SCHEMA if fixed else SCHEMA)
            or value["run_id"] != config["run_id"] or value["round_id"] != config["round_id"]
            or value["initial_checkpoint_sha256"] != config["student"]["initial_checkpoint_sha256"]):
        raise ValueError("MSE sidecar run/round/parent identity differs")
    for field in (("interval_iterations", "max_iterations") if fixed else
                  ("interval_iterations", "max_iterations", "min_validation_epochs", "patience")):
        if type(value[field]) is not int or value[field] < 1:
            raise ValueError("MSE cadence/budget/patience must be explicit positive integers")
    if not value["interval_iterations"] <= value["max_iterations"] <= config["execution"]["window_count"]:
        raise ValueError("MSE sidecar exceeds the host-bound maximum window budget")
    if fixed:
        if (value['interval_iterations'] != 100 or value['max_iterations'] != 100
                or value['full_batch_environments'] != 32 or value['rollout_steps'] != 81
                or value['checkpoint_selection'] != 'final_update_100'):
            raise ValueError('fixed100 requires exactly 100 full optimizer updates')
    else:
        for field in ("relative_improvement", "absolute_improvement"):
            if isinstance(value[field], bool) or not isinstance(value[field], (float, int)) or not math.isfinite(value[field]) or value[field] < 0:
                raise ValueError("MSE improvement thresholds must be finite and nonnegative")
        require_sha256(value["validation_manifest_sha256"], "MSE validation manifest")
        if not Path(value["validation_manifest_path"]).is_absolute():
            raise ValueError("MSE validation manifest requires an absolute shared path")
    final = value["final_test"]
    if (set(final) != {"scenario_manifest_sha256", "scenario_ids", "observation_mode"}
            or final["scenario_manifest_sha256"] != config["validation"]["scenario_manifest_sha256"]
            or final["observation_mode"] != config["validation"]["observation_mode"]
            or not isinstance(final["scenario_ids"], list) or not final["scenario_ids"]
            or any(not isinstance(item, str) or not item for item in final["scenario_ids"])
            or len(final["scenario_ids"]) != len(set(final["scenario_ids"]))):
        raise ValueError("MSE final test identity differs")
    if fixed and (len(final['scenario_ids']) != 295 or final['observation_mode'] != 'generated'):
        raise ValueError('fixed100 requires the complete rendered295 final test')
    return value


class MseBetaRoundDriver(BetaRoundDriver):
    state_filename = "mse-beta-round-state.json"

    def __init__(self, *args, mse_protocol: dict, mse_protocol_file_sha256: str, **kwargs):
        self.mse_protocol = mse_protocol
        self.mse_sha256 = hashlib.sha256(canonical(mse_protocol)).hexdigest()
        self.mse_file_sha256 = require_sha256(mse_protocol_file_sha256, "MSE sidecar file hash")
        self.selected = None
        super().__init__(*args, **kwargs)
        if (mse_protocol["run_id"] != self.config["run_id"]
                or mse_protocol["round_id"] != self.round_id
                or mse_protocol["initial_checkpoint_sha256"] != self.config["initial_checkpoint_sha256"]
                or not 1 <= mse_protocol["max_iterations"] <= self.windows):
            raise ValueError("MSE native driver sidecar identity/budget mismatch")

    def mse_status(self):
        response = self.request("GET", f"/v1/mse-rounds/{self.config['run_id']}/{self.round_id}")
        value = response.get("mse_round", {})
        selection = value.get("selection")
        if self.mse_protocol.get('schema') == FIXED100_SCHEMA:
            if (response.get('ok') is not True or value.get('protocol_sha256') != self.mse_sha256
                    or value.get('protocol') != self.mse_protocol
                    or value.get('run_id') != self.config['run_id'] or value.get('round_id') != self.round_id
                    or not isinstance(selection,dict) or set(selection) != {'checkpoint_selection','best_checkpoint_sha256','history'}
                    or selection.get('checkpoint_selection') != 'final_update_100'):
                raise RuntimeError('fixed100 coordinator protocol/identity differs')
            if value.get('status') == 'active':
                if (selection['history'] != []
                        or selection['best_checkpoint_sha256'] != self.config['initial_checkpoint_sha256']):
                    raise RuntimeError('fixed100 active selection must retain parent without validation history')
            elif value.get('status') in {'selected_pending_test','complete'}:
                selected=require_sha256(selection['best_checkpoint_sha256'],'fixed100 final checkpoint')
                if (value.get('iteration_id') != 100 or value.get('checkpoint_sha256') != selected
                        or value.get('stop_reason') != 'fixed_100_updates'
                        or selection['history'] != [{'iteration':100,'result':{'checkpoint_sha256':selected}}]):
                    raise RuntimeError('fixed100 final selection lacks exact update100 receipt')
            else:
                raise RuntimeError('fixed100 coordinator status differs')
            return value
        if (response.get("ok") is not True or value.get("protocol_sha256") != self.mse_sha256
                or value.get("protocol") != self.mse_protocol
                or value.get("run_id") != self.config["run_id"] or value.get("round_id") != self.round_id
                or not isinstance(selection, dict) or not selection.get("history")
                or selection["history"][0].get("iteration") != 0
                or selection["history"][0]["result"].get("checkpoint_sha256") != self.config["initial_checkpoint_sha256"]
                or selection.get("validation_manifest_sha256") != self.mse_protocol["validation_manifest_sha256"]):
            raise RuntimeError("MSE coordinator protocol/baseline identity mismatch")
        return value

    def status(self):
        value = super().status()
        self.mse_status()  # Before native lifecycle begin, and at each barrier.
        return value

    def wait_candidate(self, index, parent):
        deadline = self.monotonic() + self.timeout
        while self.monotonic() < deadline:
            status = super().status()
            mse = self.mse_status()
            row = next((item for item in status["iterations"] if item["round_id"] == self.round_id
                        and item["iteration_id"] == index + 1), None)
            if row and row.get("candidate_checkpoint_sha256"):
                candidate = require_sha256(row["candidate_checkpoint_sha256"], "candidate")
                barriers = [item for item in status["barriers"] if item["round_id"] == self.round_id
                            and item["iteration_id"] == index + 1]
                if (row["input_checkpoint_sha256"] != parent or candidate == parent
                        or len(barriers) != len(actor_ids(self.config))
                        or {item["actor_id"] for item in barriers} != set(actor_ids(self.config))
                        or any(item["checkpoint_sha256"] != parent or item["chunk_ids"] != [
                            f"d{self.round_id}-{item['actor_id']}-window-{index:06d}"] for item in barriers)):
                    raise RuntimeError("MSE learner candidate/all-actor barrier lineage mismatch")
                if row["status"] == "awaiting_mse_validation":
                    if self.mse_protocol.get('schema') == FIXED100_SCHEMA:
                        raise RuntimeError('fixed100 must never enter held-out MSE validation')
                    # The coordinator and MSE records are read in two requests.
                    # Selection may commit between them; re-read at the safe
                    # barrier instead of falsely declaring simulator uncertainty.
                    if mse["status"] == "selected_pending_test":
                        self.sleep(1)
                        continue
                    if status["run"]["status"] != "awaiting_mse_validation" or mse["status"] != "active":
                        raise RuntimeError("MSE validation barrier state mismatch")
                    self.sleep(1)
                    continue
                if row["status"] == "selected_pending_test":
                    selected = require_sha256(mse["selection"]["best_checkpoint_sha256"], "MSE selected checkpoint")
                    history = mse["selection"]["history"]
                    admissible = {self.config["initial_checkpoint_sha256"], candidate,
                                  *(item["output_checkpoint_sha256"] for item in self.state["windows"])}
                    if (status["run"]["status"] != "selected_pending_test" or mse["status"] != "selected_pending_test"
                            or mse["iteration_id"] != index + 1 or mse["checkpoint_sha256"] != candidate
                            or selected not in admissible
                            or selected not in {item["result"]["checkpoint_sha256"] for item in history}
                            or history[-1]["iteration"] != index + 1
                            or mse["stop_reason"] not in ({'fixed_100_updates'} if self.mse_protocol.get('schema') == FIXED100_SCHEMA else {"validation_mse_plateau", "maximum_iterations"})
                            or (self.mse_protocol.get('schema') == FIXED100_SCHEMA and (index + 1 != 100 or selected != candidate))
                            or (mse["stop_reason"] == "maximum_iterations" and index + 1 != self.mse_protocol["max_iterations"])
                            or any(item["round_id"] == self.round_id and item["iteration_id"] > index + 1 for item in status["iterations"])):
                        raise RuntimeError("MSE selected checkpoint/boundary proof mismatch")
                    self.selected = {"selected_checkpoint_sha256": selected, "stop_reason": mse["stop_reason"]}
                    return candidate
                next_row = next((item for item in status["iterations"] if item["round_id"] == self.round_id
                                 and item["iteration_id"] == index + 2), {})
                if (index + 1 >= self.mse_protocol["max_iterations"] or row["status"] != "advanced"
                        or mse["status"] != "active" or next_row.get("status") != "collecting"
                        or next_row.get("input_checkpoint_sha256") != candidate
                        or status["run"]["active_checkpoint_sha256"] != candidate):
                    raise RuntimeError("MSE candidate not safely activated before next window")
                return candidate
            self.sleep(1)
        raise TimeoutError("MSE checkpoint/validation barrier timed out")

    def should_finish(self, index, candidate):
        return self.selected is not None

    def completion_metadata(self):
        if self.selected is None:
            raise RuntimeError("MSE round exhausted budget without a selection receipt")
        return {"completion_schema": "roborender.mse-actor-completion.v1",
                "mse_protocol_sha256": self.mse_sha256,
                "mse_protocol_file_sha256": self.mse_file_sha256,
                "completed_windows": len(self.state["windows"]),
                "declared_max_iterations": self.mse_protocol["max_iterations"], **self.selected}
