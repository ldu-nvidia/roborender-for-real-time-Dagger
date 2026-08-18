import unittest

from scripts.scaling.image3f_probe import capability_error


class ProbeCapabilityTest(unittest.TestCase):
    def test_accepts_protocol_two_and_sufficient_lag(self):
        response = {
            "protocol_version": 2,
            "capabilities": {
                "lagged_reference_history": True,
                "reference_lag_max": 512,
            },
        }
        self.assertIsNone(
            capability_error(
                response,
                minimum_protocol_version=2,
                required_reference_lag=256,
                require_reference_history=True,
            )
        )

    def test_rejects_insufficient_or_malformed_lag(self):
        base = {"protocol_version": 2, "capabilities": {"lagged_reference_history": True}}
        base["capabilities"]["reference_lag_max"] = 128
        self.assertIn(
            "< required 256",
            capability_error(
                base,
                minimum_protocol_version=2,
                required_reference_lag=256,
                require_reference_history=True,
            ),
        )
        base["capabilities"]["reference_lag_max"] = "invalid"
        self.assertIn(
            "malformed",
            capability_error(
                base,
                minimum_protocol_version=2,
                required_reference_lag=256,
                require_reference_history=True,
            ),
        )


if __name__ == "__main__":
    unittest.main()
