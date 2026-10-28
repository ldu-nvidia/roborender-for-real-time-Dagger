"""Versioned offline BC recipes; Raven's three-view launcher renders one clip."""

from __future__ import annotations

INDEPENDENT_RECIPE = "raven_3view_50step_independent_v2"
LEGACY_RECIPE = "legacy_autoregressive_v1"
CURRENT_RECIPE = INDEPENDENT_RECIPE
INDEPENDENT_CONTRACT = "raven_three_view_independent_v2"
LEGACY_CONTRACT = "raven_three_view_autoregressive_v1"
RECIPE_NAMES = (INDEPENDENT_RECIPE, LEGACY_RECIPE)
COLLECTION_FIELDS = (
    "source_run_tag", "rank", "local_env_id", "global_env_id", "motion_id", "motion_sha256",
    "simulator_seed", "episode_id", "split", "request_sha256", "sampling_plan_sha256",
    "teacher_only", "selected_for_split", "overcollection",
)


def collection_provenance(item: dict) -> dict | None:
    """Retain complete fresh-cohort provenance; never invent it for old pools."""
    if not any(key in item for key in ("motion_id", "split", "sampling_plan_sha256")):
        return None
    missing = set(COLLECTION_FIELDS) - set(item)
    if missing:
        raise ValueError(f"incomplete explicit BC collection provenance: {sorted(missing)}")
    value = {key: item[key] for key in COLLECTION_FIELDS}
    if "selection_policy" in item or "emitted_chunk_ordinal" in item:
        if (item.get("selection_policy") not in {"prefix_quota_v1", "explicit_preselected_ordinals_v2", "all_complete_emissions_v1"}
                or type(item.get("emitted_chunk_ordinal")) is not int or item["emitted_chunk_ordinal"] < 0):
            raise ValueError("invalid explicit BC complete-chunk selection provenance")
        if item['selection_policy']=='all_complete_emissions_v1' and item.get('split')!='train':
            raise ValueError('continuous BC emissions are train-only')
        value.update(selection_policy=item["selection_policy"], emitted_chunk_ordinal=item["emitted_chunk_ordinal"])
    if (value["split"] not in {"train", "validation", "test"}
            or value["teacher_only"] is not True or value["selected_for_split"] is not True
            or value["overcollection"] is not False
            or not isinstance(value["episode_id"], str) or not value["episode_id"]):
        raise ValueError("unselected or ambiguous fresh BC cohort provenance")
    if (any(type(value[key]) is not int or value[key] < 0
            for key in ("rank", "local_env_id", "global_env_id", "simulator_seed"))
            or any(not isinstance(value[key], str) or not value[key]
                   for key in ("source_run_tag", "motion_id"))
            or any(not isinstance(value[key], str) or len(value[key]) != 64
                   or any(c not in "0123456789abcdef" for c in value[key])
                   for key in ("motion_sha256", "request_sha256", "sampling_plan_sha256"))):
        raise ValueError("invalid fresh BC provenance identity/hash/seed")
    return value


def recipe_config(name: str = CURRENT_RECIPE) -> dict:
    if name == INDEPENDENT_RECIPE:
        return {
            "render_recipe": name,
            "render_contract": INDEPENDENT_CONTRACT,
            "max_chain_chunks": 1,
            "matches_raven_three_view_launcher": True,
        }
    if name == LEGACY_RECIPE:
        return {
            "render_recipe": name,
            "render_contract": LEGACY_CONTRACT,
            "max_chain_chunks": 6,
            "matches_raven_three_view_launcher": False,
        }
    raise ValueError(f"unknown BC render recipe: {name!r}")


def validate_context(context: dict, *, recipe: str | None = None) -> dict:
    selected = context.get("render_recipe", CURRENT_RECIPE) if recipe is None else recipe
    config = recipe_config(selected)
    if context.get("render_recipe", selected) != selected:
        raise ValueError("job render recipe differs from the selected recipe")
    if context.get("render_contract", config["render_contract"]) != config["render_contract"]:
        raise ValueError("job render contract differs from the selected recipe")
    index, length = int(context["chain_index"]), int(context["chain_length"])
    if not 0 <= index < length <= config["max_chain_chunks"]:
        raise ValueError("chain position is incompatible with the selected render recipe")
    if bool(context["inherits_previous_reference"]) != (index > 0):
        raise ValueError("inherited-reference flag differs from the chain position")
    return config


def context_metadata(context: dict) -> dict:
    config = validate_context(context)
    result = {
        "render_contract": config["render_contract"],
        "render_chain_id": str(context["chain_id"]),
        "render_chain_index": int(context["chain_index"]),
        "render_chain_length": int(context["chain_length"]),
        "render_inherits_previous_reference": bool(context["inherits_previous_reference"]),
    }
    if config["render_recipe"] == INDEPENDENT_RECIPE:
        stream = context.get("source_stream_id")
        if not isinstance(stream, str) or not stream:
            raise ValueError("independent render requires an explicit source_stream_id")
        result["render_recipe"] = config["render_recipe"]
        result["render_source_stream_id"] = stream
    provenance = context.get("collection_provenance")
    if provenance is not None:
        validated = collection_provenance(provenance)
        if validated is None:
            raise ValueError("empty explicit BC collection provenance")
        result.update({f"collection_{key}": value for key, value in validated.items()})
    return result
