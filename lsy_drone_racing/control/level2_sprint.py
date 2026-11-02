"""Portable Level 2 controller with a smooth first-section actor transition.

The original and adapted MIT-licensed reference actors and calibration share one
policy archive. No experiment logs or private race state are needed at runtime.
See policies/level2_reference_LICENSE for the original policy's attribution.
"""

from lsy_drone_racing.control.level2_distance_actor import Level2DistanceActor


class Level2Sprint(Level2DistanceActor):
    """Use the adapted approach actor far away and the original actor near gate one."""

    POLICY_FILE = "level2_sprint.npz"
    GAINS_FILE = "level2_sprint.npz"
    TRAINED_PHASES = (0,)
    ADAPTED_PHASES = (0,)
    ADAPTED_FAR = True
    MIX_NEAR = 0.6
    MIX_FAR = 1.6
