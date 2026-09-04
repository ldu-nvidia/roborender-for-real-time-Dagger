#!/usr/bin/env python3
"""Create or reconcile the curated ``mini_prod_2`` W&B saved workspace.

This is deliberately a control-plane utility.  ``wandb-workspaces`` is loaded
only when :func:`create_saved_workspace` is called, so neither the learner nor
the actor acquires a runtime dependency on the workspace API.

The local receipt is the durable identity of the remote saved view.  A retry
loads that exact URL and reconciles it in place.  If a process dies after the
remote create but before publishing the receipt, the retained intent makes a
retry fail closed instead of creating a duplicate saved view.
"""

from __future__ import annotations

import argparse
import fcntl
import importlib
import json
import os
import re
import sys
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType
from typing import Any, Iterator
from urllib.parse import unquote, urlparse


REPOSITORY_ROOT = Path(__file__).absolute().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from round_dagger.artifacts import (  # noqa: E402
    ArtifactError,
    canonical_json_sha256,
    load_strict_json,
    publish_immutable_json,
    sha256_file,
)


WORKSPACE_INTENT_SCHEMA = "fast_roborender.wandb_workspace_intent.v1"
WORKSPACE_RECEIPT_SCHEMA = "fast_roborender.wandb_workspace_receipt.v1"
DEFAULT_WORKSPACE_NAME = "mini_prod_2"
DEFAULT_ENTITY = "nvidia"
DEFAULT_PROJECT = "lepton-dagger-production-1"
SECTION_NAMES = (
    "1_scene_samples",
    "2_training_quality",
    "3_validation_ood",
)

_IDENTITY_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")

# A declarative specification keeps the remote layout reviewable and gives the
# receipt a stable digest.  Section order here is the required workspace order.
WORKSPACE_SPEC: tuple[dict[str, Any], ...] = (
    {
        "name": "1_scene_samples",
        "panels": (
            {
                "kind": "media_browser",
                "title": "Actual conditions and RoboRender scene",
                "media_keys": ("1_scene_samples/conditions_and_rendered",),
                "num_columns": 2,
                "mode": "gallery",
            },
        ),
    },
    {
        "name": "2_training_quality",
        "panels": (
            {
                "kind": "line_plot",
                "title": "Training action error",
                "x": "optimizer_step",
                "y": (
                    "2_training_quality/action_mse",
                    "2_training_quality/action_mae",
                    "2_training_quality/action_rmse",
                ),
            },
            {
                "kind": "line_plot",
                "title": "Training agreement and tail error",
                "x": "optimizer_step",
                "y": (
                    "2_training_quality/action_cosine",
                    "2_training_quality/action_absolute_error_p90",
                    "2_training_quality/action_l2_p90",
                ),
            },
            {
                "kind": "line_plot",
                "title": "Optimization health",
                "x": "optimizer_step",
                "y": (
                    "2_training_quality/optimization_loss",
                    "2_training_quality/gradient_l2_norm",
                    "2_training_quality/learning_rate",
                ),
            },
            {
                "kind": "line_plot",
                "title": "DAgger supervision and intervention",
                "x": "optimizer_step",
                "y": (
                    "2_training_quality/expert_intervention_rate",
                    "2_training_quality/behavior_expert_match_rate",
                    "2_training_quality/expert_label_coverage",
                    "2_training_quality/scheduled_beta",
                ),
            },
            {
                "kind": "line_plot",
                "title": "Replay and sample coverage",
                "x": "optimizer_step",
                "y": (
                    "2_training_quality/global_samples_seen",
                    "2_training_quality/replay_transitions",
                    "2_training_quality/replay_environments",
                    "2_training_quality/probe_transitions",
                    "2_training_quality/done_rate",
                ),
            },
        ),
    },
    {
        "name": "3_validation_ood",
        "panels": (
            {
                "kind": "line_plot",
                "title": "OOD held-out action error",
                "x": "optimizer_step",
                "y": (
                    "3_validation_ood/action_mse",
                    "3_validation_ood/action_mae",
                    "3_validation_ood/action_rmse",
                ),
            },
            {
                "kind": "line_plot",
                "title": "OOD agreement and tail error",
                "x": "optimizer_step",
                "y": (
                    "3_validation_ood/action_cosine",
                    "3_validation_ood/action_absolute_error_p90",
                    "3_validation_ood/action_l2_p90",
                ),
            },
            {
                "kind": "line_plot",
                "title": "OOD improvement from step zero",
                "x": "optimizer_step",
                "y": (
                    "3_validation_ood/action_mse_delta_from_step0",
                    "3_validation_ood/action_mse_relative_improvement_from_step0",
                ),
            },
            {
                "kind": "line_plot",
                "title": "OOD evaluation coverage",
                "x": "optimizer_step",
                "y": (
                    "3_validation_ood/transitions",
                    "3_validation_ood/eval_frontier",
                ),
            },
        ),
    },
)


class WorkspaceControlError(RuntimeError):
    """The saved workspace cannot be created or reconciled safely."""


def _absolute(path: str | os.PathLike[str]) -> Path:
    return Path(os.path.abspath(os.path.expanduser(os.fspath(path))))


def _validate_identity(value: str, label: str) -> str:
    if not isinstance(value, str) or _IDENTITY_RE.fullmatch(value) is None:
        raise WorkspaceControlError(f"{label} is not a safe W&B identity")
    return value


def _validate_workspace_name(value: str) -> str:
    # The name is also used in the runset filter.  Restricting it to the same
    # token grammar as W&B identities avoids expression quoting ambiguity.
    if not isinstance(value, str) or _IDENTITY_RE.fullmatch(value) is None:
        raise WorkspaceControlError("workspace_name is unsafe")
    return value


def _spec_document(workspace_name: str) -> dict[str, Any]:
    """Return a JSON-shaped copy of the immutable workspace specification."""

    return {
        "auto_generate_panels": False,
        "run_filter": f"Name = '{workspace_name}'",
        "sections": json.loads(json.dumps(WORKSPACE_SPEC)),
    }


def _load_workspace_modules() -> tuple[ModuleType, ModuleType]:
    try:
        workspaces = importlib.import_module("wandb_workspaces.workspaces")
        reports = importlib.import_module("wandb_workspaces.reports.v2")
    except (ImportError, ModuleNotFoundError) as exc:
        raise WorkspaceControlError(
            "the optional wandb-workspaces package is required only for this "
            "control-plane command; install it before creating the saved workspace"
        ) from exc
    return workspaces, reports


def build_sections(workspaces: ModuleType | Any, reports: ModuleType | Any) -> list[Any]:
    """Materialize the exact three ordered W&B sections from ``WORKSPACE_SPEC``."""

    sections: list[Any] = []
    for section_spec in WORKSPACE_SPEC:
        panels: list[Any] = []
        for panel_spec in section_spec["panels"]:
            if panel_spec["kind"] == "media_browser":
                panels.append(
                    reports.MediaBrowser(
                        title=panel_spec["title"],
                        media_keys=list(panel_spec["media_keys"]),
                        num_columns=panel_spec["num_columns"],
                        mode=panel_spec["mode"],
                    )
                )
            elif panel_spec["kind"] == "line_plot":
                panels.append(
                    reports.LinePlot(
                        title=panel_spec["title"],
                        x=panel_spec["x"],
                        y=list(panel_spec["y"]),
                    )
                )
            else:  # pragma: no cover - constant guarded for future edits.
                raise WorkspaceControlError(
                    f"unsupported workspace panel kind: {panel_spec['kind']}"
                )
        sections.append(
            workspaces.Section(
                name=section_spec["name"],
                panels=panels,
                is_open=True,
            )
        )
    observed = tuple(getattr(section, "name", None) for section in sections)
    if observed != SECTION_NAMES:
        raise WorkspaceControlError("workspace section construction changed order")
    return sections


def _intent_path(receipt_path: Path) -> Path:
    return receipt_path.with_name(f".{receipt_path.name}.intent.json")


def _lock_path(receipt_path: Path) -> Path:
    return receipt_path.with_name(f".{receipt_path.name}.lock")


@contextmanager
def _exclusive_state_lock(receipt_path: Path) -> Iterator[None]:
    parent = receipt_path.parent
    parent.mkdir(parents=True, exist_ok=True)
    try:
        if parent.resolve(strict=True) != parent:
            raise WorkspaceControlError("workspace state directory contains a symlink")
    except OSError as exc:
        raise WorkspaceControlError("cannot resolve workspace state directory") from exc
    flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0) | getattr(
        os, "O_NOFOLLOW", 0
    )
    try:
        descriptor = os.open(_lock_path(receipt_path), flags, 0o600)
    except OSError as exc:
        raise WorkspaceControlError("cannot open workspace state lock") from exc
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _expected_intent(
    *, entity: str, project: str, workspace_name: str
) -> dict[str, Any]:
    return {
        "schema": WORKSPACE_INTENT_SCHEMA,
        "entity": entity,
        "project": project,
        "workspace_name": workspace_name,
        "section_names": list(SECTION_NAMES),
        "workspace_spec_sha256": canonical_json_sha256(
            _spec_document(workspace_name)
        ),
    }


def _load_exact(path: Path, expected: Mapping[str, Any], label: str) -> dict[str, Any]:
    try:
        document = load_strict_json(path)
    except ArtifactError as exc:
        raise WorkspaceControlError(f"{label} is unsafe or invalid") from exc
    if not isinstance(document, Mapping) or dict(document) != dict(expected):
        raise WorkspaceControlError(f"{label} does not match this workspace request")
    return dict(document)


def _validate_workspace_url(url: Any, *, entity: str, project: str) -> str:
    if not isinstance(url, str) or len(url) > 2048:
        raise WorkspaceControlError("W&B did not return a saved workspace URL")
    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.hostname not in {"wandb.ai", "www.wandb.ai"}:
        raise WorkspaceControlError("saved workspace URL is not an HTTPS wandb.ai URL")
    if parsed.username is not None or parsed.password is not None or parsed.fragment:
        raise WorkspaceControlError("saved workspace URL contains forbidden credentials or fragment")
    components = [unquote(component) for component in parsed.path.split("/") if component]
    if len(components) < 2 or components[:2] != [entity, project]:
        raise WorkspaceControlError("saved workspace URL targets a different entity or project")
    return url


def _expected_receipt(
    *,
    intent_sha256: str,
    entity: str,
    project: str,
    workspace_name: str,
    workspace_url: str,
) -> dict[str, Any]:
    return {
        "schema": WORKSPACE_RECEIPT_SCHEMA,
        "intent_sha256": intent_sha256,
        "entity": entity,
        "project": project,
        "workspace_name": workspace_name,
        "workspace_url": workspace_url,
        "section_names": list(SECTION_NAMES),
        "workspace_spec_sha256": canonical_json_sha256(
            _spec_document(workspace_name)
        ),
    }


def _load_receipt(
    path: Path,
    *,
    intent_sha256: str,
    entity: str,
    project: str,
    workspace_name: str,
) -> dict[str, Any]:
    try:
        document = load_strict_json(path)
    except ArtifactError as exc:
        raise WorkspaceControlError("workspace receipt is unsafe or invalid") from exc
    if not isinstance(document, Mapping):
        raise WorkspaceControlError("workspace receipt must be a JSON object")
    url = _validate_workspace_url(
        document.get("workspace_url"), entity=entity, project=project
    )
    expected = _expected_receipt(
        intent_sha256=intent_sha256,
        entity=entity,
        project=project,
        workspace_name=workspace_name,
        workspace_url=url,
    )
    if dict(document) != expected:
        raise WorkspaceControlError("workspace receipt does not match this workspace request")
    return dict(document)


def _validate_loaded_identity(workspace: Any, *, entity: str, project: str) -> None:
    for field, expected in (("entity", entity), ("project", project)):
        observed = getattr(workspace, field, None)
        if observed is not None and observed != expected:
            raise WorkspaceControlError(f"loaded workspace {field} identity changed")
    auto_panels = getattr(workspace, "auto_generate_panels", False)
    if auto_panels is not False:
        raise WorkspaceControlError(
            "existing workspace auto-generates panels and cannot satisfy the exact layout"
        )


def create_saved_workspace(
    *,
    entity: str,
    project: str,
    workspace_name: str,
    receipt_path: str | os.PathLike[str],
    workspace_module: ModuleType | Any | None = None,
    reports_module: ModuleType | Any | None = None,
) -> dict[str, Any]:
    """Create once, or reconcile the receipt-pinned saved view in place.

    The returned document is also stored immutably at ``receipt_path``.  Both
    optional module arguments exist for focused unit tests; production callers
    should leave them unset.
    """

    entity = _validate_identity(entity, "entity")
    project = _validate_identity(project, "project")
    workspace_name = _validate_workspace_name(workspace_name)
    if (workspace_module is None) != (reports_module is None):
        raise WorkspaceControlError(
            "workspace_module and reports_module must be supplied together"
        )
    receipt = _absolute(receipt_path)
    intent = _intent_path(receipt)
    expected_intent = _expected_intent(
        entity=entity, project=project, workspace_name=workspace_name
    )

    with _exclusive_state_lock(receipt):
        intent_preexisted = os.path.lexists(intent)
        if intent_preexisted:
            _load_exact(intent, expected_intent, "workspace intent")

        if os.path.lexists(receipt):
            if not intent_preexisted:
                raise WorkspaceControlError("workspace receipt exists without its intent")
            intent_sha256 = sha256_file(intent)
            receipt_document = _load_receipt(
                receipt,
                intent_sha256=intent_sha256,
                entity=entity,
                project=project,
                workspace_name=workspace_name,
            )
            if workspace_module is None or reports_module is None:
                workspace_module, reports_module = _load_workspace_modules()
            sections = build_sections(workspace_module, reports_module)
            try:
                workspace = workspace_module.Workspace.from_url(
                    receipt_document["workspace_url"]
                )
                _validate_loaded_identity(workspace, entity=entity, project=project)
                workspace.name = workspace_name
                workspace.sections = sections
                workspace.runset_settings = workspace_module.RunsetSettings(
                    filters=f"Name = '{workspace_name}'"
                )
                workspace.save()
            except WorkspaceControlError:
                raise
            except Exception as exc:
                raise WorkspaceControlError(
                    "cannot reconcile the receipt-pinned W&B saved workspace"
                ) from exc
            observed_url = _validate_workspace_url(
                getattr(workspace, "url", receipt_document["workspace_url"]),
                entity=entity,
                project=project,
            )
            if observed_url != receipt_document["workspace_url"]:
                raise WorkspaceControlError("workspace reconciliation changed saved-view identity")
            return receipt_document

        if intent_preexisted:
            raise WorkspaceControlError(
                "workspace intent exists without a receipt; a prior remote create is "
                "ambiguous, so refusing to create a duplicate saved view"
            )

        if workspace_module is None or reports_module is None:
            workspace_module, reports_module = _load_workspace_modules()
        sections = build_sections(workspace_module, reports_module)
        runset_settings = workspace_module.RunsetSettings(
            filters=f"Name = '{workspace_name}'"
        )
        # Publish the intent immediately before the first remote mutation.  All
        # local imports and panel construction have already succeeded, so an
        # orphaned intent means the create truly is remotely ambiguous.
        try:
            publish_immutable_json(intent, expected_intent)
        except ArtifactError as exc:
            raise WorkspaceControlError("cannot publish workspace intent") from exc
        intent_sha256 = sha256_file(intent)
        try:
            workspace = workspace_module.Workspace(
                name=workspace_name,
                entity=entity,
                project=project,
                sections=sections,
                runset_settings=runset_settings,
                auto_generate_panels=False,
            )
            workspace.save_as_new_view()
        except Exception as exc:
            raise WorkspaceControlError("cannot create W&B saved workspace") from exc
        workspace_url = _validate_workspace_url(
            getattr(workspace, "url", None), entity=entity, project=project
        )
        receipt_document = _expected_receipt(
            intent_sha256=intent_sha256,
            entity=entity,
            project=project,
            workspace_name=workspace_name,
            workspace_url=workspace_url,
        )
        try:
            publish_immutable_json(receipt, receipt_document)
        except ArtifactError as exc:
            raise WorkspaceControlError("cannot publish workspace receipt") from exc
        return receipt_document


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entity", default=DEFAULT_ENTITY)
    parser.add_argument("--project", default=DEFAULT_PROJECT)
    parser.add_argument("--workspace-name", default=DEFAULT_WORKSPACE_NAME)
    parser.add_argument(
        "--receipt",
        required=True,
        help="durable local JSON receipt that pins the remote saved-view URL",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        receipt = create_saved_workspace(
            entity=arguments.entity,
            project=arguments.project,
            workspace_name=arguments.workspace_name,
            receipt_path=arguments.receipt,
        )
    except WorkspaceControlError as exc:
        print(f"workspace error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
