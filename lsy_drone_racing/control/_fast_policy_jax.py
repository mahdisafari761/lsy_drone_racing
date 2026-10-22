"""Public observation encoding and deterministic JAX racing actor."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.scipy.spatial.transform import Rotation


def features(obs: dict, history: jax.Array, actions: jax.Array) -> jax.Array:
    """Encode public actor features plus the observed sequence phase."""
    position = obs["pos"]
    rotation = Rotation.from_quat(obs["quat"]).as_matrix()
    batch = jnp.arange(position.shape[0])
    corners = jnp.array([[0, -0.2, 0.2], [0, 0.2, 0.2], [0, 0.2, -0.2], [0, -0.2, -0.2]])
    outputs = []
    for advance in [0, 1]:
        index = jnp.minimum(obs["n_gates_passed"] + advance, obs["gate_sequence"].shape[-1] - 1)
        gid = obs["gate_sequence"][batch, index]
        gate_rotation = Rotation.from_quat(obs["gates_quat"][batch, gid]).as_matrix()
        direction = obs["gate_sequence_direction"][batch, index]
        local = (
            jnp.broadcast_to(corners, (position.shape[0], 4, 3))
            .at[:, :, 1]
            .multiply(direction[:, None])
        )
        world = (
            jnp.einsum("nij,nkj->nki", gate_rotation, local) + obs["gates_pos"][batch, gid, None, :]
        )
        body = jnp.einsum("nji,nkj->nki", rotation, world - position[:, None, :])
        outputs.append(body.reshape(position.shape[0], 12))
    poles = (
        obs["obstacles_pos"]
        .at[:, :, 2]
        .set(jnp.minimum(obs["obstacles_pos"][:, :, 2], position[:, None, 2]))
    )
    relative = jnp.einsum("nji,nkj->nki", rotation, poles - position[:, None, :])
    order = jnp.argsort(jnp.linalg.norm(relative, axis=-1), axis=-1)
    relative = jnp.take_along_axis(relative, order[:, :, None], axis=1).reshape(
        position.shape[0], 12
    )
    velocity = jnp.einsum("nji,nj->ni", rotation, obs["vel"])
    return jnp.concatenate(
        [
            position[:, 2:3],
            velocity,
            obs["ang_vel"],
            rotation.reshape(-1, 9),
            *outputs,
            actions[:, 0, :],
            relative,
            history.reshape(-1, 32),
            jax.nn.one_hot(jnp.minimum(obs["n_gates_passed"], 4), 5),
            obs["gates_visited"].astype(jnp.float32),
        ],
        axis=-1,
    )


def basic_state(obs: dict) -> jax.Array:
    """Encode the historical world-frame state."""
    matrix = Rotation.from_quat(obs["quat"]).as_matrix().reshape(-1, 9)
    return jnp.concatenate([obs["pos"][:, 2:3], matrix, obs["vel"], obs["ang_vel"]], axis=-1)


def actor(params: dict, x: jax.Array) -> jax.Array:
    """Compute a pretrained mean plus optional phase-specific residual control."""
    phase = (
        jnp.argmax(x[:, 88:93], axis=-1)
        if x.shape[-1] > 88
        else jnp.zeros(x.shape[0], dtype=jnp.int32)
    )
    extra = x[:, 88:]
    raw = x[:, :88]
    x = raw
    hidden = None
    for index, (weight, bias) in enumerate(params["actor"]):
        x = jnp.tanh(x @ weight.T + bias)
        if index == 1:
            hidden = x
    if "residual" in params or "phase_residual" in params:
        encoded = jnp.concatenate([raw, hidden], axis=-1)
        base = x
        if "unfrozen" not in params:
            encoded = jax.lax.stop_gradient(encoded)
            base = jax.lax.stop_gradient(base)
        if "phase_residual" in params:
            weight, bias = params["phase_residual"]
            delta = jnp.einsum("nij,nj->ni", weight[phase], encoded) + bias[phase]
        else:
            weight, bias = params["residual"]
            delta = encoded @ weight.T + bias
        x = base + 0.5 * jnp.tanh(delta)
    if "initial_actor" in params:

        def initial_policy(_: None) -> jax.Array:
            initial = raw
            initial_hidden = None
            for index, (weight, bias) in enumerate(params["initial_actor"]):
                initial = jnp.tanh(initial @ weight.T + bias)
                if index == 1:
                    initial_hidden = initial
            weight, bias = params["initial_residual"]
            return initial + 0.5 * jnp.tanh(
                jnp.concatenate([raw, initial_hidden], axis=-1) @ weight.T + bias
            )

        initial = jax.lax.cond(
            jnp.any(phase == 0), initial_policy, lambda _: jnp.zeros_like(x), None
        )
        x = jnp.where((phase == 0)[:, None], initial, x)
    if "mlp" in params:
        encoded = jax.lax.stop_gradient(jnp.concatenate([raw, hidden, extra], axis=-1))
        base = jax.lax.stop_gradient(x)

        def correction(_: None) -> jax.Array:
            if params["mlp"][0][0].ndim == 3 and encoded.shape[0] >= 64:
                # Dense grouped products avoid materializing one weight matrix per sample.
                result = jnp.zeros_like(x)
                for head in range(params["mlp"][0][0].shape[0]):
                    head_result = encoded
                    for weight, bias in params["mlp"]:
                        head_result = jnp.tanh(head_result @ weight[head].T + bias[head])
                    result += jnp.where((phase == head)[:, None], head_result, 0.0)
                return result
            result = encoded
            for weight, bias in params["mlp"]:
                if weight.ndim == 3:
                    result = jnp.tanh(jnp.einsum("nij,nj->ni", weight[phase], result) + bias[phase])
                else:
                    result = jnp.tanh(result @ weight.T + bias)
            return result

        if "mlp_phase" in params:
            mask = (phase == params["mlp_phase"])[:, None]
            delta = jax.lax.cond(jnp.any(mask), correction, lambda _: jnp.zeros_like(x), None)
            x = base + 0.5 * delta * mask
        else:
            x = base + 0.5 * correction(None)
    return x
