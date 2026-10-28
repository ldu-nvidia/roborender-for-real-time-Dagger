"""New launches disable MSE validation; historical execution is explicit only.

This gate does not admit a runtime, qualify capacity, or alter frozen protocols.
Call it before creating an execution intent or contacting a native owner.
"""


def require_training_launch(*, validation_enabled, allow_legacy_validation=False):
    if type(validation_enabled) is not bool or type(allow_legacy_validation) is not bool:
        raise ValueError('explicit boolean training validation policy required')
    if validation_enabled and not allow_legacy_validation:
        raise ValueError(
            'MSE validation is disabled for new runs. This is a historical '
            'validation-based workflow, not the new fixed-final BC/fixed100 '
            'DAgger warm pipeline. Historical execution requires explicit '
            '--allow-legacy-validation compatibility opt-in.')
