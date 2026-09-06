from __future__ import annotations

import os
import subprocess
import unittest

from round_dagger.learner_identity import (
    learner_identity_sha256,
)


class LearnerIdentityTest(unittest.TestCase):
    def _inputs(self) -> dict[str, object]:
        return {
            "campaign_id": "semantic-campaign",
            "macro_round": 7,
            "learner_code_sha256": "1" * 64,
            "learner_config_sha256": "2" * 64,
            "replay_snapshot_file_sha256": "3" * 64,
            "replay_snapshot_sha256": "4" * 64,
            "parent_checkpoint_sha256": "5" * 64,
            "parent_runtime_policy_sha256": "6" * 64,
            "runtime_container_sha256": "7" * 64,
            "learner_contract_sha256": "8" * 64,
        }

    def test_exact_retry_reuses_identity_namespace(self) -> None:
        inputs = self._inputs()
        first = learner_identity_sha256(**inputs)
        retry = learner_identity_sha256(**inputs)
        self.assertEqual(first, retry)
        self.assertEqual(f"identity_{first}", f"identity_{retry}")
        self.assertEqual(f"rr-rd-{first}", f"rr-rd-{retry}")

    def test_snapshot_parent_runtime_container_and_contract_each_separate(self) -> None:
        inputs = self._inputs()
        baseline = learner_identity_sha256(**inputs)
        mutations = {
            "campaign_id": "other-campaign",
            "macro_round": 8,
            "learner_code_sha256": "0" * 64,
            "learner_config_sha256": "f" * 64,
            "replay_snapshot_file_sha256": "9" * 64,
            "replay_snapshot_sha256": "a" * 64,
            "parent_checkpoint_sha256": "b" * 64,
            "parent_runtime_policy_sha256": "c" * 64,
            "runtime_container_sha256": "d" * 64,
            "learner_contract_sha256": "e" * 64,
        }
        for field, value in mutations.items():
            with self.subTest(field=field):
                changed = dict(inputs)
                changed[field] = value
                self.assertNotEqual(baseline, learner_identity_sha256(**changed))

    def test_python_identity_matches_shell_boundary_encoding(self) -> None:
        inputs = self._inputs()
        expected = learner_identity_sha256(**inputs)
        environment = dict(os.environ)
        environment.update({key.upper(): str(value) for key, value in inputs.items()})
        script = r"""
printf '%s\n' \
  'fast_roborender.round_dagger.learner_run_identity.v2' \
  "campaign_id=$CAMPAIGN_ID" \
  "macro_round=$MACRO_ROUND" \
  "learner_code_sha256=$LEARNER_CODE_SHA256" \
  "learner_config_sha256=$LEARNER_CONFIG_SHA256" \
  "replay_snapshot_file_sha256=$REPLAY_SNAPSHOT_FILE_SHA256" \
  "replay_snapshot_sha256=$REPLAY_SNAPSHOT_SHA256" \
  "parent_checkpoint_sha256=$PARENT_CHECKPOINT_SHA256" \
  "parent_runtime_policy_sha256=$PARENT_RUNTIME_POLICY_SHA256" \
  "runtime_container_sha256=$RUNTIME_CONTAINER_SHA256" \
  "learner_contract_sha256=$LEARNER_CONTRACT_SHA256" | \
  sha256sum | awk '{print $1}'
"""
        completed = subprocess.run(
            ["bash", "-c", script],
            env=environment,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(completed.stdout.strip(), expected)

    def test_maximum_campaign_token_is_accepted(self) -> None:
        inputs = self._inputs()
        inputs["campaign_id"] = "c" * 128
        identity = learner_identity_sha256(**inputs)
        self.assertRegex(identity, r"[0-9a-f]{64}")


if __name__ == "__main__":
    unittest.main()
