"""Batched public-observation implementation of the attributed reference policy."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.spatial.transform import Rotation

if TYPE_CHECKING:
    from pathlib import Path


def load_policy(path: Path) -> dict:
    """Load plain arrays without executing objects from a policy archive."""
    with np.load(path, allow_pickle=False) as archive:
        return {key: jnp.asarray(archive[key]) for key in archive.files}


def reference_features(obs: dict, previous: jax.Array) -> jax.Array:
    """Encode all 46 reference inputs from observable states and object poses."""
    position = obs["pos"]
    rotation = Rotation.from_quat(obs["quat"]).as_matrix()
    yaw = jnp.arctan2(rotation[:, 1, 0], rotation[:, 0, 0])
    cosine, sine = jnp.cos(yaw), jnp.sin(yaw)
    batch = jnp.arange(position.shape[0])

    def heading(vector: jax.Array) -> jax.Array:
        return jnp.stack(
            [
                cosine * vector[:, 0] + sine * vector[:, 1],
                -sine * vector[:, 0] + cosine * vector[:, 1],
                vector[:, 2],
            ],
            axis=-1,
        )

    def gate(advance: int) -> jax.Array:
        slot = jnp.minimum(obs["n_gates_passed"] + advance, obs["gate_sequence"].shape[-1] - 1)
        index = obs["gate_sequence"][batch, slot]
        sign = obs["gate_sequence_direction"][batch, slot, None]
        matrix = Rotation.from_quat(obs["gates_quat"][batch, index]).as_matrix()
        return jnp.concatenate(
            [
                heading(obs["gates_pos"][batch, index] - position),
                heading(matrix[:, :, 0] * sign),
                heading(matrix[:, :, 1] * sign),
                obs["gates_visited"][batch, index, None],
            ],
            axis=-1,
        )

    second = obs["n_gates_passed"] + 1 < obs["gate_sequence"].shape[-1]
    poles = obs["obstacles_pos"][:, :, :2] - position[:, None, :2]
    order = jnp.argsort(jnp.sum(poles * poles, axis=-1), axis=-1)[:, :2]
    near = jnp.take_along_axis(poles, order[:, :, None], axis=1)
    x = cosine[:, None] * near[:, :, 0] + sine[:, None] * near[:, :, 1]
    y = -sine[:, None] * near[:, :, 0] + cosine[:, None] * near[:, :, 1]
    visited = jnp.take_along_axis(obs["obstacles_visited"], order, axis=1)
    pillars = jnp.stack([x, y, visited], axis=-1).reshape(position.shape[0], 6)
    return jnp.concatenate(
        [
            heading(obs["vel"]),
            obs["ang_vel"],
            heading(rotation[:, :, 0]),
            heading(rotation[:, :, 2]),
            sine[:, None],
            cosine[:, None],
            position[:, 2:3],
            gate(0),
            gate(1) * second[:, None],
            second[:, None],
            pillars,
            previous,
        ],
        axis=-1,
    )


def actor(params: dict, encoded: jax.Array) -> jax.Array:
    """Evaluate the ELU mean network in its original normalized feature space."""
    normalized = jnp.clip((encoded[:, :46] - params["obs_mean"]) / params["obs_std"], -10.0, 10.0)
    value = normalized
    for index in range(4):
        value = value @ params[f"W{index}"] + params[f"b{index}"]
        if index < 3:
            value = jax.nn.elu(value)
    value = jnp.clip(value, -1.0, 1.0)
    if "ref_mlp.0.weight" in params:
        phase = jnp.argmax(encoded[:, 46:51], axis=-1)
        correction = jnp.concatenate([normalized, encoded[:, 46:51]], axis=-1)
        for index in range(2):
            weight, bias = params[f"ref_mlp.{index}.weight"], params[f"ref_mlp.{index}.bias"]
            correction = jnp.tanh(jnp.einsum("nij,nj->ni", weight[phase], correction) + bias[phase])
        value = value + 0.25 * correction
    return jnp.clip(value, -1.0, 1.0)
