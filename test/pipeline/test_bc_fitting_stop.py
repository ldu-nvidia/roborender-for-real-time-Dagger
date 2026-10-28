import unittest

from pipeline.bc_training.stopping_gate import should_stop_for_plateau, training_target_met
from pipeline.bc_training.stopping_gate import fitting_stop_contract, validate_fitting_stop_resume, epoch_sequence
from itertools import islice


class FittingStopTest(unittest.TestCase):
    def test_disabled_preserves_plateau_rule(self):
        self.assertTrue(should_stop_for_plateau(True, None, None))
        self.assertFalse(should_stop_for_plateau(False, None, None))

    def test_both_conditions_required(self):
        self.assertFalse(should_stop_for_plateau(True, .02, .0714))
        self.assertFalse(should_stop_for_plateau(False, .02, .019))
        self.assertTrue(should_stop_for_plateau(True, .02, .019))

    def test_strict_threshold(self):
        self.assertFalse(training_target_met(.02, .02))
        self.assertTrue(training_target_met(.02, 0.0))

    def test_bad_target_rejected(self):
        for target in (0, -.1, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                training_target_met(target, .01)

    def test_missing_or_nonfinite_measurement_rejected(self):
        for measured in (None, -.1, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                should_stop_for_plateau(False, .02, measured)

    def test_epoch_goal_is_distinct_from_frozen_eval_goal(self):
        contract = fitting_stop_contract(epoch_target=.015)
        self.assertIn("full_epoch_training_mode", contract["measurement"])
        self.assertNotEqual(contract, fitting_stop_contract(eval_target=.015))
        with self.assertRaises(ValueError):
            fitting_stop_contract(.015, .015)

    def test_epoch_goal_strict_and_needs_validation_plateau(self):
        self.assertFalse(should_stop_for_plateau(True, .015, .015))
        self.assertFalse(should_stop_for_plateau(False, .015, .0149))
        self.assertTrue(should_stop_for_plateau(True, .015, .0149))

    def test_uncapped_really_passes_old_ceiling(self):
        self.assertEqual(list(islice(epoch_sequence(99, None), 5)), [100,101,102,103,104])
        self.assertEqual(list(islice(epoch_sequence(9999, None), 2)), [10000,10001])
        self.assertEqual(list(epoch_sequence(98, 100)), [99,100])
        with self.assertRaises(ValueError): epoch_sequence(100, 100)

    def test_resume_admits_explicit_disabled_to_goal_only(self):
        target = fitting_stop_contract(epoch_target=.015)
        validate_fitting_stop_resume(None, target)
        validate_fitting_stop_resume(fitting_stop_contract(), target)
        validate_fitting_stop_resume(target, target)
        for different in (fitting_stop_contract(epoch_target=.02), fitting_stop_contract(eval_target=.015)):
            with self.assertRaises(ValueError): validate_fitting_stop_resume(different, target)


if __name__ == '__main__':
    unittest.main()
