"""Benchmark complete races on reproducible, independent Level 2 seeds.

Example (from the repository root)::

    pixi run python scripts/benchmark_level2.py --controller level2_predictive_85.py \
        --n-runs 100 --seed-start 10000 --output logs/level2-validation.json

The track, physics, disturbances and randomization distributions are loaded unchanged.
Only rendering, the controller, its permitted control interface and RNG seeds are selected.
Reported race time matches ``scripts/sim.py``: the first environment step is indexed at 0 s.
The JSON also includes the elapsed simulation time, one step longer, to make this convention
explicit. Controllers receive only the public observation, info and configuration.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import time
from pathlib import Path
from typing import Any

import gymnasium
import numpy as np
from gymnasium.wrappers.jax_to_numpy import JaxToNumpy

from lsy_drone_racing.utils import load_config, load_controller

LOGGER = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[1]


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> list[float]:
    """Return the two-sided 95% Wilson binomial confidence interval."""
    if total <= 0:
        return [0.0, 1.0]
    proportion = successes / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(proportion * (1 - proportion) / total + z * z / (4 * total**2))
    radius /= denominator
    return [max(0.0, center - radius), min(1.0, center + radius)]


def distribution(values: list[float]) -> dict[str, float | None]:
    """Summarize a sequence, preserving missing values for an empty sample."""
    if not values:
        return {name: None for name in ("mean", "median", "p95", "max")}
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
    }


def summarize(episodes: list[dict], target_time: float, target_success: float) -> dict[str, Any]:
    """Report successful-race time separately from the success fraction."""
    successful = [episode["official_time_s"] for episode in episodes if episode["success"]]
    count, successes = len(episodes), len(successful)
    success_rate = successes / count if count else 0.0
    times = distribution(successful)
    return {
        "n_runs": count,
        "successes": successes,
        "failures": count - successes,
        "success_rate": success_rate,
        "success_rate_wilson_95": wilson_interval(successes, count),
        "successful_official_time_s": times,
        "point_estimate_meets_target": bool(
            times["mean"] is not None
            and times["mean"] <= target_time
            and success_rate >= target_success
        ),
        "target_mean_time_s": target_time,
        "target_success_rate": target_success,
    }


def benchmark(
    controller: str | None = None,
    n_runs: int = 100,
    seed_start: int = 10000,
    seeds: list[int] | None = None,
    output: Path | None = None,
    control_mode: str | None = None,
    trace: bool = False,
    target_time: float = 3.2,
    target_success: float = 0.8,
    controller_settings: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run full races, optionally saving public trajectories for failure diagnosis."""
    selected_seeds = list(range(seed_start, seed_start + n_runs)) if seeds is None else seeds
    if not selected_seeds or len(set(selected_seeds)) != len(selected_seeds):
        raise ValueError("Provide a nonempty list of distinct seeds")
    if any(seed < 0 or seed >= 2**32 for seed in selected_seeds):
        raise ValueError("Seeds must be unsigned 32-bit integers")
    if target_time <= 0 or not 0 < target_success <= 1:
        raise ValueError("Targets require positive time and success rate in (0, 1]")
    config_path = ROOT / "config/level2.toml"
    config = load_config(config_path)
    config.sim.render = False
    if control_mode is not None:
        if control_mode not in ("attitude", "state"):
            raise ValueError("control_mode must be attitude or state")
        config.env.control_mode = control_mode
    controller_path = Path(controller or config.controller.file)
    if not controller_path.is_absolute():
        controller_path = ROOT / "lsy_drone_racing/control" / controller_path
    controller_cls = load_controller(controller_path)
    settings = controller_settings or {}
    for name, value in settings.items():
        if name.startswith("_") or not hasattr(controller_cls, name):
            raise ValueError(f"Unknown public controller setting: {name}")
        if callable(getattr(controller_cls, name)):
            raise ValueError(f"Cannot override controller method: {name}")
        setattr(controller_cls, name, value)
    artifacts = {}
    for name in ("POLICY_FILE", "GAINS_FILE"):
        filename = getattr(controller_cls, name, None)
        if isinstance(filename, str):
            artifact_path = controller_path.parent / "policies" / filename
            if artifact_path.is_file():
                artifacts[name] = {
                    "path": str(artifact_path),
                    "sha256": hashlib.sha256(artifact_path.read_bytes()).hexdigest(),
                }
    env = JaxToNumpy(
        gymnasium.make(
            config.env.id,
            freq=config.env.freq,
            sim_config=config.sim,
            sensor_range=config.env.sensor_range,
            control_mode=config.env.control_mode,
            track=config.env.track,
            disturbances=config.env.get("disturbances"),
            randomizations=config.env.get("randomizations"),
            seed=selected_seeds[0],
        )
    )
    result: dict[str, Any] = {
        "controller": str(controller_path),
        "controller_sha256": hashlib.sha256(controller_path.read_bytes()).hexdigest(),
        "config": str(config_path),
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "control_mode": config.env.control_mode,
        "controller_settings": settings,
        "controller_artifacts": artifacts,
        "seeds": selected_seeds,
        "time_convention": "official_time_s=(steps-1)/freq; elapsed_simulation_s=steps/freq",
        "episodes": [],
    }
    all_compute: list[float] = []
    all_loop: list[float] = []
    start = time.perf_counter()
    try:
        for seed in selected_seeds:
            # Expose the chosen seed through the public configuration, as in the official script.
            config.env.seed = seed
            obs, info = env.reset(seed=seed)
            init_start = time.perf_counter()
            ctrl = controller_cls(obs, info, config)
            init_seconds = time.perf_counter() - init_start
            compute_times, loop_times, gate_times, public_trace = [], [], [], []
            steps, passed = 0, 0
            while True:
                before = time.perf_counter()
                action = ctrl.compute_control(obs, info)
                compute_seconds = time.perf_counter() - before
                obs, reward, terminated, truncated, info = env.step(action)
                callback_start = time.perf_counter()
                controller_finished = ctrl.step_callback(
                    action, obs, reward, terminated, truncated, info
                )
                callback_seconds = time.perf_counter() - callback_start
                compute_times.append(compute_seconds)
                loop_times.append(compute_seconds + callback_seconds)
                current_time = steps / config.env.freq
                steps += 1
                new_passed = int(obs["n_gates_passed"])
                gate_times.extend([current_time] * (new_passed - passed))
                passed = new_passed
                if trace:
                    public_trace.append(
                        {
                            "t": current_time,
                            "pos": np.asarray(obs["pos"]).tolist(),
                            "vel": np.asarray(obs["vel"]).tolist(),
                            "quat": np.asarray(obs["quat"]).tolist(),
                            "action": np.asarray(action).tolist(),
                            "n_gates_passed": passed,
                            "gates_pos": np.asarray(obs["gates_pos"]).tolist(),
                            "gates_quat": np.asarray(obs["gates_quat"]).tolist(),
                            "gates_visited": np.asarray(obs["gates_visited"]).tolist(),
                            "obstacles_pos": np.asarray(obs["obstacles_pos"]).tolist(),
                            "obstacles_visited": np.asarray(obs["obstacles_visited"]).tolist(),
                        }
                    )
                if terminated or truncated or controller_finished:
                    break
            success = passed == obs["gate_sequence"].shape[0]
            ctrl.episode_callback()
            ctrl.episode_reset()
            episode = {
                "seed": seed,
                "success": bool(success),
                "official_time_s": current_time,
                "elapsed_simulation_s": steps / config.env.freq,
                "gates_passed": passed,
                "gate_pass_times_s": gate_times,
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "controller_finished": bool(controller_finished),
                "final_pos": np.asarray(obs["pos"]).tolist(),
                "final_vel": np.asarray(obs["vel"]).tolist(),
                "initialization_s": init_seconds,
                "compute_s": distribution(compute_times),
                "compute_plus_callback_s": distribution(loop_times),
                "control_budget_exceedances": sum(x > 1 / config.env.freq for x in loop_times),
            }
            if trace:
                episode["trace"] = public_trace
            result["episodes"].append(episode)
            all_compute.extend(compute_times)
            all_loop.extend(loop_times)
            result["summary"] = summarize(result["episodes"], target_time, target_success)
            if output is not None:
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
            LOGGER.info(
                "seed=%d success=%s time=%.2fs gates=%d init=%.2fs loop_p95=%.4fs",
                seed,
                success,
                current_time,
                passed,
                init_seconds,
                episode["compute_plus_callback_s"]["p95"],
            )
    finally:
        env.close()
    result["wall_time_s"] = time.perf_counter() - start
    result["compute_s"] = distribution(all_compute)
    result["compute_plus_callback_s"] = distribution(all_loop)
    result["control_budget_s"] = 1 / config.env.freq
    result["control_budget_exceedances"] = sum(x > 1 / config.env.freq for x in all_loop)
    if output is not None:
        output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    return result


def main():
    """Parse benchmark arguments and print the compact summary."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controller")
    parser.add_argument("--n-runs", type=int, default=100)
    parser.add_argument("--seed-start", type=int, default=10000)
    parser.add_argument("--seeds", help="Comma-separated explicit seed list; overrides n-runs")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--control-mode", choices=("state", "attitude"))
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--target-time", type=float, default=3.2)
    parser.add_argument("--target-success", type=float, default=0.8)
    parser.add_argument(
        "--controller-settings",
        type=json.loads,
        help='JSON object of existing public class attributes, e.g. {"RATE_GAIN": 2.0}',
    )
    args = vars(parser.parse_args())
    if args["seeds"] is not None:
        args["seeds"] = [int(seed.strip()) for seed in args["seeds"].split(",")]
    logging.basicConfig(level=logging.INFO)
    result = benchmark(**args)
    print(json.dumps(result["summary"], indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
