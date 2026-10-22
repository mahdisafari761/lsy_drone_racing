"""Invert the public Mellinger state-to-attitude map, including its thrust scaling."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation


def invert_geometry(
    position: Any,
    velocity: Any,
    desired: Any,
    current: Any,
    thrust: Any,
    rates: Any,
    params: dict,
    xp: Any,
) -> Any:
    """Construct a zero-position-error state command for a desired attitude/thrust."""
    axis = desired[..., :, 2]
    alignment = xp.maximum(xp.sum(axis * current[..., :, 2], axis=-1), 0.2)
    scale = params["mass_thrust"] * params["thrust_max"] * 4 / params["pwm_max"]
    magnitude = thrust / (scale * alignment)
    acceleration = axis * magnitude[..., None] / params["mass"] + params["gravity_vec"]
    # Mellinger projects a horizontal heading vector onto the desired body XY plane.
    heading = (
        desired[..., :, 0] - axis * (desired[..., 2, 0] / xp.maximum(axis[..., 2], 0.1))[..., None]
    )
    yaw = xp.arctan2(heading[..., 1], heading[..., 0])
    zero = xp.zeros_like(yaw)
    quaternion = xp.stack([zero, zero, xp.sin(yaw / 2), xp.cos(yaw / 2)], axis=-1)
    return xp.concatenate([position, velocity, acceleration, quaternion, rates], axis=-1)


def attitude_to_state(
    obs: dict, command: np.ndarray, rates: np.ndarray, params: dict
) -> np.ndarray:
    """Convert an attitude command without changing its requested collective thrust."""
    desired = Rotation.from_euler("xyz", command[:3]).as_matrix()
    current = Rotation.from_quat(obs["quat"]).as_matrix()
    return invert_geometry(
        np.asarray(obs["pos"]),
        np.asarray(obs["vel"]),
        desired,
        current,
        np.asarray(command[3]),
        rates,
        params,
        np,
    ).astype(np.float32)
