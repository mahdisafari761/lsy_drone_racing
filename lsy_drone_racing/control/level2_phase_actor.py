"""Choose a fine-tuned racing actor by section while retaining Lead calibration.

Both actors descend from the MIT-licensed public reference documented in
level2_reference.py. Their weights are packaged together so a policy archive
checksum covers the complete deployed actor selection.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from lsy_drone_racing.control.level2_lead import Level2Lead


class Level2PhaseActor(Level2Lead):
    """Use one applied-action history across both cached actor networks."""

    POLICY_FILE = "level2_phase_actor.npz"
    TRAINED_PHASES = (1, 2, 3)

    def __init__(self, obs: dict, info: dict, config: dict):
        """Load the base and fine-tuned layers from one frozen policy archive."""
        super().__init__(obs, info, config)
        phases = tuple(self.TRAINED_PHASES)
        if any(not isinstance(phase, (int, np.integer)) or not 0 <= phase < 5 for phase in phases):
            raise ValueError("TRAINED_PHASES must contain section indices from 0 to 4")
        self.trained_phases = frozenset(phases)
        self.base_layers = self.layers
        with np.load(
            Path(__file__).parent / "policies" / self.POLICY_FILE, allow_pickle=False
        ) as archive:
            self.trained_layers = [
                (
                    archive[f"trained_W{index}"].astype(np.float64),
                    archive[f"trained_b{index}"].astype(np.float64),
                )
                for index in range(len(self.base_layers))
            ]
        for base, trained in zip(self.base_layers, self.trained_layers, strict=True):
            for original, adapted in zip(base, trained, strict=True):
                if adapted.shape != original.shape or not np.all(np.isfinite(adapted)):
                    raise ValueError("Fine-tuned layers must be finite and match the base shapes")

    def compute_control(self, obs: dict, info: dict | None = None) -> np.ndarray:
        """Select cached layers before applying the shared calibrated control path."""
        phase = min(int(obs["n_gates_passed"]), 4)
        self.layers = self.trained_layers if phase in self.trained_phases else self.base_layers
        return super().compute_control(obs, info)
