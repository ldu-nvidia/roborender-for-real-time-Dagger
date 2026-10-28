"""Explicit observation policy for newly admitted rendered-only DAgger runs.

This opt-in does not relabel historical mixed-observation runs. Isaac camera
products remain renderer controls; only Image3F output may enter the student.
"""
from __future__ import annotations

import json

GENERATED_ONLY_CONTRACT = {
    "schema": "roborender.dagger.student-observation.v1",
    "mode": "generated_only",
    "renderer": "Image3F",
    "renderer_steps": 5,
    "renderer_cfg": 1.0,
    "teacache_threshold": 0.3,
    "raytraced_policy_fallback": False,
}


def validate_generated_only(value):
    # Canonical equality also rejects booleans masquerading as numeric settings.
    if (not isinstance(value, dict)
            or json.dumps(value, sort_keys=True, allow_nan=False)
            != json.dumps(GENERATED_ONLY_CONTRACT, sort_keys=True, allow_nan=False)):
        raise ValueError("exact versioned Image3F generated_only observation contract required")
    return value


def configure_native_observations(trainer, contract):
    """Set source sampling before native episode binding, including phase reuse."""
    if not hasattr(trainer, "_roborender_original_raytraced_mix_prob"):
        trainer._roborender_original_raytraced_mix_prob = trainer.vg_raytraced_mix_prob
    if contract is None:
        trainer.vg_raytraced_mix_prob = trainer._roborender_original_raytraced_mix_prob
        trainer._roborender_student_observation_contract = None
        return
    validate_generated_only(contract)
    trainer.vg_raytraced_mix_prob = 0.0
    trainer._roborender_student_observation_contract = dict(contract)


def require_generated_source(contract, source):
    if contract is None:
        return
    validate_generated_only(contract)
    if source != "generated_controls":
        raise ValueError("generated_only student observations forbid native ray-traced RGB")
