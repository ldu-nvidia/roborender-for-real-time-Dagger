"""Tests for external, post-seal Lepton backend contract finalization."""

from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

from round_dagger import lepton_job
from scripts.round_dagger import materialize_finalized_lepton_backend_contract as finalizer


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "configs/round_dagger/production_learner_1x8h200_lepton.json"


class FinalizedLeptonBackendContractTest(unittest.TestCase):
    def test_finalizer_never_writes_bytecode_into_its_sealed_source(self) -> None:
        self.assertTrue(finalizer.sys.dont_write_bytecode)

    def test_substitution_does_not_mutate_source_inventoried_template(self) -> None:
        template = json.loads(TEMPLATE.read_text())
        finalized = finalizer.finalized_document(
            template,
            source_inventory_sha256="a" * 64,
            python_freeze_sha256="b" * 64,
        )
        self.assertEqual(
            template["supply_chain"]["source_inventory_sha256"], "0" * 64
        )
        self.assertEqual(
            finalized["supply_chain"]["source_inventory_sha256"], "a" * 64
        )
        self.assertEqual(
            finalized["supply_chain"]["python_freeze_sha256"], "b" * 64
        )
        restored = json.loads(json.dumps(finalized))
        restored["supply_chain"]["source_inventory_sha256"] = "0" * 64
        restored["supply_chain"]["python_freeze_sha256"] = "0" * 64
        self.assertEqual(restored, template)

    def test_finalizer_verifies_evidence_before_external_create_only_publish(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            project = Path(temporary) / "project"
            source = project / "src/sealed"
            source.mkdir(parents=True)
            inventory = project / "src/sealed.inventory.json"
            inventory.write_text('{"sealed":true}\n')
            inventory.chmod(0o444)
            environment = project / "envs/h200_learner.freeze.txt"
            environment.parent.mkdir()
            environment.write_text("example==1\n")
            environment.chmod(0o444)
            template = source / "backend-template.json"
            template.write_bytes(TEMPLATE.read_bytes())
            template.chmod(0o444)
            output = project / "manifests/lepton-backend-final.json"

            base = lepton_job.load_contract(TEMPLATE)
            contract = replace(
                base,
                project_root=project,
                source_root=source,
                source_inventory=inventory,
                python_freeze=environment,
                source_inventory_sha256="0" * 64,
                python_freeze_sha256="0" * 64,
            )
            inventory_sha = lepton_job.sha256_file(inventory)
            environment_sha = lepton_job.sha256_file(environment)
            events: list[str] = []

            def load(path, *, immutable=False):
                self.assertTrue(immutable)
                candidate = Path(path)
                if candidate == template:
                    return contract
                self.assertEqual(candidate, output)
                document = json.loads(output.read_text())
                return replace(
                    contract,
                    sha256=lepton_job.sha256_file(output),
                    document=document,
                    source_inventory_sha256=inventory_sha,
                    python_freeze_sha256=environment_sha,
                )

            def source_gate(verified):
                self.assertFalse(output.exists())
                self.assertEqual(verified.source_inventory_sha256, inventory_sha)
                events.append("source")

            def freeze_gate(verified):
                self.assertFalse(output.exists())
                self.assertEqual(verified.python_freeze_sha256, environment_sha)
                events.append("freeze")

            with (
                mock.patch.object(finalizer, "load_contract", side_effect=load),
                mock.patch.object(
                    finalizer, "verify_sealed_contract_source", side_effect=source_gate
                ),
                mock.patch.object(
                    finalizer,
                    "verify_python_environment_freeze",
                    side_effect=freeze_gate,
                ),
            ):
                result = finalizer.finalize_contract(
                    template_path=template, output_path=output
                )
            self.assertEqual(events, ["source", "freeze"])
            self.assertEqual(output.stat().st_mode & 0o777, 0o444)
            self.assertEqual(result.source_inventory_sha256, inventory_sha)
            self.assertEqual(result.python_freeze_sha256, environment_sha)
            document = json.loads(output.read_text())
            self.assertEqual(
                document["supply_chain"]["source_inventory_sha256"], inventory_sha
            )
            self.assertEqual(
                document["supply_chain"]["python_freeze_sha256"], environment_sha
            )
            with self.assertRaisesRegex(
                lepton_job.LeptonJobError, "destination already exists"
            ):
                with mock.patch.object(finalizer, "load_contract", return_value=contract):
                    finalizer.finalize_contract(
                        template_path=template, output_path=output
                    )


if __name__ == "__main__":
    unittest.main()
