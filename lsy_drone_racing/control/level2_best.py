"""Frozen best extensively measured tradeoff from the current Level 2 experiments.

Development seeds 30000-30199: 158/200 successes, mean successful time
3.374556962 seconds. The requested <=3.30 s and >80% target is not met.
The bundled actors descend from the MIT-licensed public reference; see
policies/level2_reference_LICENSE and policies/README.md for provenance.
"""

from lsy_drone_racing.control.level2_phase_actor import Level2PhaseActor


class Level2Best(Level2PhaseActor):
    """Use the 1M actor for the first two sections and the 400k actor afterwards."""

    POLICY_FILE = "level2_best.npz"
    GAINS_FILE = "level2_best_gains.npz"
    TRAINED_PHASES = (0, 1)
