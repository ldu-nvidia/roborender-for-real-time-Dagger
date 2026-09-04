from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from round_dagger.artifacts import publish_immutable_json
from scripts.round_dagger import create_mini_prod_2_workspace as dashboard


class _Panel:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class _Reports:
    class MediaBrowser(_Panel):
        pass

    class LinePlot(_Panel):
        pass


class _Section:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class _RunsetSettings:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class _Workspace:
    created: list["_Workspace"] = []
    remote: dict[str, "_Workspace"] = {}
    new_view_calls = 0
    save_calls = 0
    from_url_calls: list[str] = []

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        self.url = None
        self.__class__.created.append(self)

    @classmethod
    def reset(cls) -> None:
        cls.created = []
        cls.remote = {}
        cls.new_view_calls = 0
        cls.save_calls = 0
        cls.from_url_calls = []

    @classmethod
    def from_url(cls, url: str):
        cls.from_url_calls.append(url)
        return cls.remote[url]

    def save_as_new_view(self) -> None:
        self.__class__.new_view_calls += 1
        self.url = (
            f"https://wandb.ai/{self.entity}/{self.project}/workspace"
            "?nw=mini-prod-2-saved-view"
        )
        self.__class__.remote[self.url] = self

    def save(self) -> None:
        self.__class__.save_calls += 1


class _Workspaces:
    Section = _Section
    RunsetSettings = _RunsetSettings
    Workspace = _Workspace


class MiniProd2WorkspaceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="mini_prod_2_workspace_")
        self.receipt = Path(self.temporary.name) / "state" / "workspace.json"
        _Workspace.reset()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def create(self, **overrides):
        arguments = {
            "entity": "nvidia",
            "project": "lepton-dagger-production-1",
            "workspace_name": "mini_prod_2",
            "receipt_path": self.receipt,
            "workspace_module": _Workspaces,
            "reports_module": _Reports,
        }
        arguments.update(overrides)
        return dashboard.create_saved_workspace(**arguments)

    def test_creates_exact_three_ordered_curated_sections(self) -> None:
        receipt = self.create()

        self.assertEqual(_Workspace.new_view_calls, 1)
        self.assertEqual(_Workspace.save_calls, 0)
        self.assertEqual(receipt["section_names"], list(dashboard.SECTION_NAMES))
        self.assertTrue(self.receipt.is_file())
        workspace = _Workspace.created[0]
        self.assertFalse(workspace.auto_generate_panels)
        self.assertEqual(workspace.runset_settings.filters, "Name = 'mini_prod_2'")
        self.assertEqual(
            [section.name for section in workspace.sections],
            ["1_scene_samples", "2_training_quality", "3_validation_ood"],
        )
        self.assertEqual([len(section.panels) for section in workspace.sections], [1, 5, 4])
        media = workspace.sections[0].panels[0]
        self.assertIsInstance(media, _Reports.MediaBrowser)
        self.assertEqual(
            media.media_keys, ["1_scene_samples/conditions_and_rendered"]
        )
        training_keys = {
            key
            for panel in workspace.sections[1].panels
            for key in panel.y
        }
        self.assertIn("2_training_quality/action_mse", training_keys)
        self.assertIn("2_training_quality/expert_intervention_rate", training_keys)
        self.assertIn("2_training_quality/gradient_l2_norm", training_keys)
        validation_keys = {
            key
            for panel in workspace.sections[2].panels
            for key in panel.y
        }
        self.assertIn("3_validation_ood/action_mse", validation_keys)
        self.assertIn(
            "3_validation_ood/action_mse_relative_improvement_from_step0",
            validation_keys,
        )
        self.assertTrue(
            all(key.startswith("3_validation_ood/") for key in validation_keys)
        )

    def test_retry_reconciles_same_saved_view_without_creating_duplicate(self) -> None:
        first = self.create()
        workspace = _Workspace.remote[first["workspace_url"]]
        workspace.name = "tampered"
        workspace.sections = []

        second = self.create()

        self.assertEqual(second, first)
        self.assertEqual(_Workspace.new_view_calls, 1)
        self.assertEqual(_Workspace.save_calls, 1)
        self.assertEqual(_Workspace.from_url_calls, [first["workspace_url"]])
        self.assertEqual(workspace.name, "mini_prod_2")
        self.assertEqual(workspace.runset_settings.filters, "Name = 'mini_prod_2'")
        self.assertEqual(
            [section.name for section in workspace.sections],
            list(dashboard.SECTION_NAMES),
        )

    def test_orphaned_intent_fails_closed_instead_of_duplicating_view(self) -> None:
        intent_path = dashboard._intent_path(self.receipt)
        publish_immutable_json(
            intent_path,
            dashboard._expected_intent(
                entity="nvidia",
                project="lepton-dagger-production-1",
                workspace_name="mini_prod_2",
            ),
        )

        with self.assertRaisesRegex(
            dashboard.WorkspaceControlError, "refusing to create a duplicate"
        ):
            self.create()
        self.assertEqual(_Workspace.new_view_calls, 0)
        self.assertFalse(self.receipt.exists())

    def test_receipt_identity_mismatch_is_rejected_before_remote_update(self) -> None:
        self.create()
        with self.assertRaisesRegex(
            dashboard.WorkspaceControlError, "intent does not match"
        ):
            self.create(project="different-project")
        self.assertEqual(_Workspace.new_view_calls, 1)
        self.assertEqual(_Workspace.save_calls, 0)

    def test_missing_optional_package_has_actionable_error(self) -> None:
        with mock.patch.object(
            dashboard.importlib,
            "import_module",
            side_effect=ModuleNotFoundError("wandb_workspaces"),
        ):
            with self.assertRaisesRegex(
                dashboard.WorkspaceControlError, "optional wandb-workspaces"
            ):
                dashboard.create_saved_workspace(
                    entity="nvidia",
                    project="lepton-dagger-production-1",
                    workspace_name="mini_prod_2",
                    receipt_path=self.receipt,
                )
        # Local dependency failure happens before any remote-mutation intent.
        self.assertFalse(dashboard._intent_path(self.receipt).exists())
        self.assertFalse(self.receipt.exists())

    def test_existing_auto_panel_workspace_cannot_claim_exact_layout(self) -> None:
        receipt = self.create()
        workspace = _Workspace.remote[receipt["workspace_url"]]
        workspace.auto_generate_panels = True
        with self.assertRaisesRegex(
            dashboard.WorkspaceControlError, "auto-generates panels"
        ):
            self.create()
        self.assertEqual(_Workspace.save_calls, 0)


if __name__ == "__main__":
    unittest.main()
