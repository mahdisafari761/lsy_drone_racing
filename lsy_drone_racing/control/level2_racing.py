"""Faster gate-aligned MPC retaining observation-driven collision-aware routing."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.spatial.transform import Rotation

from lsy_drone_racing.control.level2_mpc import Level2MPC

if TYPE_CHECKING:
    from ml_collections import ConfigDict


class Level2Racing(Level2MPC):
    """Use faster tracking, rolling gate alignment and local collision braking.

    The barrier projection is a heuristic safeguard for the configured vehicle,
    not a formal safety guarantee under arbitrary model or tracking errors.
    """

    def __init__(self, obs: dict, info: dict, config: ConfigDict):
        """Initialize the observation-driven planner and predictive controller."""
        super().__init__(obs, info, config)
        weights = np.tile([1000.0, 3.0, 0.03], self.n)
        weights[-3:] *= 2
        differences = np.eye(self.n) - np.eye(self.n, k=-1)
        self.GQ = self.G.T * weights
        self.H = self.GQ @ self.G + 0.08 * np.eye(self.n) + 0.15 * differences.T @ differences
        self.invH = np.linalg.inv(self.H)
        self.L = np.linalg.eigvalsh(self.H)[-1]

    def geometry(self, obs: dict, index: int) -> tuple[np.ndarray, np.ndarray]:
        """Choose a small aperture offset if a pole blocks the centerline."""
        center, normal = super().geometry(obs, index)
        gid = int(obs["gate_sequence"][index])
        if not obs["gates_visited"][gid]:
            return center, normal
        lateral = self.rot[gid][:, 1]
        vertical = self.rot[gid][:, 2]
        offsets = [
            np.zeros(3),
            0.05 * lateral,
            -0.05 * lateral,
            0.075 * lateral,
            -0.075 * lateral,
            0.04 * vertical,
            -0.04 * vertical,
        ]
        for offset in offsets:
            crossing = center + offset
            if self.clear_segments(crossing - 0.14 * normal, crossing + 0.14 * normal)[0]:
                return crossing, normal
        return center, normal

    def navigation(self, obs: dict, changed: bool) -> tuple[np.ndarray, float]:
        """Take off, observe and align, cross the center, then clear the gate plane."""
        pos = np.asarray(obs["pos"], float)
        passed = int(obs["n_gates_passed"])
        if passed >= len(obs["gate_sequence"]):
            return pos, 0.0
        reset_route = changed
        if passed != self.passed:
            center, normal = self.geometry(obs, self.passed)
            self.target = center + 0.28 * normal
            self.passed = passed
            self.phase = "exit"
            self.route = [self.target.copy()]
        if self.phase == "takeoff":
            if np.linalg.norm(pos - self.target) < 0.10:
                self.phase = "approach"
                reset_route = True
            else:
                return self.target, 1.8
        if self.phase == "exit":
            center, normal = self.geometry(obs, self.passed - 1)
            clearance = 0.18
            for pole in self.poles:
                offset = pole - center
                along = np.dot(offset, normal)
                if along > 0 and np.linalg.norm(offset - along * normal) < 0.24:
                    clearance = min(clearance, max(0.10, (along - 0.14) * 0.7))
            if np.dot(pos - center, normal) < clearance:
                self.target = center + min(0.24, clearance + 0.06) * normal
                return self.target, 0.8
            self.phase = "approach"
            reset_route = True
        center, normal = self.geometry(obs, passed)
        if self.phase == "approach":
            gid = int(obs["gate_sequence"][passed])
            ready = False
            if obs["gates_visited"][gid]:
                # Choose a staging distance that also clears poles near the opening.
                entries = [center - d * normal for d in [0.23, 0.20, 0.17, 0.14, 0.28, 0.34]]
                feasible = [p for p in entries if self.clear_segments(p, center)[0]]
                self.target = feasible[0] if feasible else entries[0]
                ready = bool(feasible)
            else:
                self.target = center - 0.28 * normal
            if not self.clear_segments(self.target, self.target)[0]:
                lateral = self.rot[gid][:, 1]
                candidates = [center - 0.30 * normal + s * 0.27 * lateral for s in [-1, 1]]
                candidates += [center + 0.30 * normal + s * 0.27 * lateral for s in [-1, 1]]
                feasible = [p for p in candidates if self.clear_segments(p, p)[0]]
                if feasible:
                    self.target = min(feasible, key=lambda p: np.linalg.norm(pos - p))
                    ready = False
            target_changed = self.route_goal is None or not np.allclose(
                self.target, self.route_goal, atol=0.01
            )
            blocked = (
                bool(self.route) and not self.clear_segments(pos, self.route[0], escape=True)[0]
            )
            incomplete = bool(self.route) and np.linalg.norm(self.route[-1] - self.target) > 0.01
            if (
                reset_route
                or target_changed
                or not self.route
                or ((blocked or incomplete) and self.tick >= self.next_plan_tick)
            ):
                self.route = self.plan(pos, self.target)
                self.route_goal = self.target.copy()
                self.next_plan_tick = self.tick + max(1, round(0.5 / self.dt))
            while len(self.route) > 1 and np.linalg.norm(pos - self.route[0]) < 0.06:
                self.route.pop(0)
            velocity = np.asarray(obs["vel"], float)
            signed = np.dot(pos - center, normal)
            lateral_error = pos - center - signed * normal
            lateral_velocity = velocity - np.dot(velocity, normal) * normal
            crossing_time = max(0.0, -signed) / max(0.4, np.dot(velocity, normal))
            predicted = lateral_error + lateral_velocity * min(0.5, crossing_time + 0.1)
            rolling = (
                -0.5 < signed < -0.08
                and np.linalg.norm(lateral_error) < 0.05
                and np.linalg.norm(predicted) < 0.05
                and self.clear_segments(pos, center)[0]
            )
            if ready and (
                rolling
                or (np.linalg.norm(pos - self.target) < 0.07 and np.linalg.norm(velocity) < 0.4)
            ):
                self.phase = "cross"
                self.route = []
            else:
                return self.route[0], 2.6
        self.target = center + 0.28 * normal
        return self.target, 1.8

    def compute_control(self, obs: dict, info: dict | None = None) -> np.ndarray:
        """Return [roll, pitch, yaw, collective thrust], in radians and newtons."""
        changed = self.update_map(obs)
        goal, speed = self.navigation(obs, changed)
        pos, vel = np.asarray(obs["pos"], float), np.asarray(obs["vel"], float)
        measured = (vel - self.last_vel) / self.dt
        self.last_vel = vel.copy()
        self.accel = 0.7 * self.accel + 0.3 * measured
        zaxis = Rotation.from_quat(obs["quat"]).as_matrix()[:, 2]
        # Estimate mass using observed vertical impulse and the previous applied thrust.
        if self.tick > 20 and pos[2] > 0.25 and abs(measured[2]) < 5 and zaxis[2] > 0.85:
            estimate = self.last_action[3] * zaxis[2] / max(4.0, 9.81 + measured[2])
            self.mass = 0.98 * self.mass + 0.02 * np.clip(
                estimate, 0.75 * self.mass0, 1.25 * self.mass0
            )
        delta = goal - pos
        distance = np.linalg.norm(delta)
        direction = delta / max(distance, 1e-8)
        # Build position/velocity references, braking as the waypoint approaches.
        ref = np.zeros((self.n, 3, 3))
        state = np.stack([pos, vel, self.accel])
        limits = np.array([7.0, 7.0, 5.0])
        # Limit requested speed by a reachable position reference over the horizon.
        vref = direction * min(speed, 6.0 * distance)
        tt = np.arange(1, self.n + 1) * self.h
        travel = np.minimum(np.linalg.norm(vref) * tt, distance)
        ref[:, 0, :] = pos + travel[:, None] * direction
        ref[:, 1, :] = vref
        ref[travel >= distance, 1, :] = 0
        acceleration = self.solve_mpc(state, ref, limits)
        acceleration = self.filter_acceleration(pos, vel, acceleration, obs)
        force = acceleration + np.array([0.0, 0.0, 9.81])
        pitch = np.arctan2(force[0], force[2])
        roll = np.arctan2(-force[1], np.hypot(force[0], force[2]))
        thrust = self.mass * np.dot(force, zaxis)
        action = np.array(
            [
                np.clip(roll, -0.65, 0.65),
                np.clip(pitch, -0.65, 0.65),
                0.0,
                np.clip(thrust, *self.thrust_bounds),
            ],
            dtype=np.float32,
        )
        self.last_action = action.copy()
        self.tick += 1
        return action

    def filter_acceleration(
        self, pos: np.ndarray, vel: np.ndarray, acceleration: np.ndarray, obs: dict
    ) -> np.ndarray:
        """Project acceleration onto pole and gate-frame braking constraints.

        A second-order distance barrier accounts for closing speed, including a
        short attitude-response reserve. It uses observed positions exclusively;
        unobserved poles retain their nominal-position uncertainty margin.
        """
        result = acceleration.copy()
        normals, bounds = [], []
        for pole, seen in zip(self.poles, obs["obstacles_visited"]):
            if pos[2] > pole[2] + 0.15:
                continue
            delta = pos[:2] - pole[:2]
            distance = np.linalg.norm(delta)
            if distance > 1.0:
                continue
            normal = delta / max(distance, 1e-6)
            closing = np.dot(normal, vel[:2])
            radius = 0.14 if seen else 0.32
            clearance = distance - radius + min(0.0, closing) * 0.10
            tangential_squared = max(0.0, np.dot(vel[:2], vel[:2]) - closing**2)
            bound = -8.0 * closing - 16.0 * clearance - tangential_squared / max(distance, 0.05)
            normals.append(normal)
            bounds.append(bound)
        box_specs = [
            ([0, 0, 0.28], [0.01, 0.36, 0.08]),
            ([0, 0, -0.28], [0.01, 0.36, 0.08]),
            ([0, -0.28, 0], [0.01, 0.08, 0.36]),
            ([0, 0.28, 0], [0.01, 0.08, 0.36]),
            ([0, 0, -0.86], [0.05, 0.05, 0.5]),
        ]
        # Frame barriers retain the full 3D normal, including vertical clearance.
        normals = [np.r_[normal, 0.0] for normal in normals]
        for center, rotation, seen in zip(self.centers, self.rot, obs["gates_visited"]):
            if np.linalg.norm(pos - center) > 1.2:
                continue
            for offset, half in box_specs:
                local = (pos - center) @ rotation - np.array(offset)
                outside = local - np.clip(local, -np.array(half), np.array(half))
                distance = np.linalg.norm(outside)
                if distance < 1e-6 or distance > 0.65:
                    continue
                normal = rotation @ (outside / distance)
                closing = np.dot(normal, vel)
                radius = 0.10 if seen else 0.24
                clearance = distance - radius + min(0.0, closing) * 0.08
                local_vel = vel @ rotation
                active = np.abs(outside) > 1e-8
                curvature = max(0.0, np.sum((local_vel * active) ** 2) - closing**2) / distance
                normals.append(normal)
                bounds.append(-10.0 * closing - 25.0 * clearance - curvature)
        for _ in range(6):
            for normal, bound in zip(normals, bounds):
                deficit = bound - np.dot(normal, result)
                if deficit > 0:
                    result += min(deficit, 10.0) * normal
            result = np.clip(result, [-7.0, -7.0, -5.0], [7.0, 7.0, 5.0])
        return result
