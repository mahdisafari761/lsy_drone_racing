"""Tune the frozen Level2Lead on larger, fixed development race sets.

Only calibration arrays are written under logs/lead_v2. This script never changes
the selected controller, original race configuration or deployed policy files.
Each candidate runs complete original Level 2 races from the normal platform.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from scripts.search_reference_policy import ROOT, build_rollout


def summary(completed: np.ndarray, times: np.ndarray, passed: np.ndarray) -> dict:
    """Summarize complete-race reliability and successful-race time separately."""
    successful = times[completed]
    return {
        "n_runs": int(completed.size),
        "successes": int(completed.sum()),
        "success": float(completed.mean()),
        "mean": float(successful.mean()) if successful.size else None,
        "p95": float(np.percentile(successful, 95)) if successful.size else None,
        "gates": float(passed.mean()),
    }


def rank(record: dict, minimum_success: float) -> float:
    """Favor fast reliable races while keeping failed-run times out of the mean."""
    return (record["mean"] or 10.0) + 15 * max(0.0, minimum_success - record["success"])


def main():
    """Run a narrow CEM search on fixed development worlds and hold out validation worlds."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--population", type=int, default=8)
    parser.add_argument("--trials", type=int, default=128)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--seeds", default="43000,44009")
    parser.add_argument("--validation-seeds", default="73000,74009")
    parser.add_argument("--minimum-success", type=float, default=0.86)
    parser.add_argument("--output", default="logs/lead_v2/tuning")
    parser.add_argument("--random-seed", type=int, default=20261028)
    parser.add_argument("--resume")
    args = parser.parse_args()
    development = [int(x) for x in args.seeds.split(",")]
    validation = [int(x) for x in args.validation_seeds.split(",")]
    if any(500000 <= x < 500500 for x in development + validation):
        raise ValueError("These seeds are reserved for final root-agent validation")
    base_path = ROOT / "lsy_drone_racing/control/policies/level2_lead_gains.npz"
    with np.load(base_path, allow_pickle=False) as archive:
        baseline = archive["gains"].astype(np.float64)
    mean = baseline.copy()
    if args.resume:
        with np.load(args.resume, allow_pickle=False) as archive:
            mean = archive["gains"].astype(np.float64)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    env, rollout = build_rollout(args.population, args.trials, lead_control=True)
    rng = np.random.default_rng(args.random_seed)
    deviation = np.tile([0.012, 0.035, 0.0025, 0.016], (5, 1))
    floor = np.array([0.003, 0.010, 0.0007, 0.004])
    minimum = np.maximum(baseline - [0.06, 0.12, 0.012, 0.08], [0.88, 0.0, -0.03, 0.85])
    maximum = np.minimum(baseline + [0.06, 0.20, 0.012, 0.08], [1.15, 0.55, 0.03, 1.18])
    history = {
        "baseline": baseline.tolist(),
        "development_seeds": development,
        "validation_seeds": validation,
        "trials_per_seed": args.trials,
        "minimum_success": args.minimum_success,
        "iterations": [],
    }
    best_gains, best_score = mean.copy(), float("inf")
    total_start = time.perf_counter()

    def evaluate(gains: np.ndarray, seeds: list[int]) -> list[dict]:
        success, times, passed = [], [], []
        for seed in seeds:
            c, t, p = map(
                np.asarray,
                jax.device_get(rollout(jnp.asarray(gains, jnp.float32), jnp.asarray(seed))),
            )
            success.append(c)
            times.append(t)
            passed.append(p)
        success = np.concatenate(success, axis=1)
        times = np.concatenate(times, axis=1)
        passed = np.concatenate(passed, axis=1)
        return [summary(c, t, p) for c, t, p in zip(success, times, passed, strict=True)]

    try:
        for iteration in range(args.iterations):
            candidates = np.clip(
                mean + rng.normal(size=(args.population, 5, 4)) * deviation, minimum, maximum
            )
            candidates[0], candidates[1], candidates[2] = baseline, mean, best_gains
            if iteration == 0 and args.population >= 8:
                # First compare restrained global changes instead of twenty-dimensional jumps.
                candidates[3:] = baseline
                candidates[3, :, 0] += 0.01
                candidates[4, :, 0] -= 0.01
                candidates[5, :, 2] += 0.003
                candidates[6, :, 2] -= 0.003
                candidates[7, :, 1] *= 0.5
            start = time.perf_counter()
            records = evaluate(candidates, development)
            scores = np.asarray([rank(r, args.minimum_success) for r in records])
            elite = np.argsort(scores)[: max(2, args.population // 4)]
            selected = int(elite[0])
            if scores[selected] < best_score:
                best_score, best_gains = float(scores[selected]), candidates[selected].copy()
            for index, (gains, record) in enumerate(zip(candidates, records, strict=True)):
                record.update(index=index, score=float(scores[index]))
                np.savez(
                    output.parent / f"candidate-{iteration:03d}-{index:02d}.npz",
                    gains=gains.astype(np.float32),
                )
            iteration_record = {
                "iteration": iteration,
                "seconds": time.perf_counter() - start,
                "records": records,
                "gains": candidates.tolist(),
                "selected": selected,
            }
            history["iterations"].append(iteration_record)
            history.update(best_score=best_score, best_gains=best_gains.tolist())
            output.with_suffix(".json").write_text(json.dumps(history, indent=2), encoding="utf-8")
            np.savez(output.with_suffix(".npz"), gains=best_gains.astype(np.float32))
            print(
                json.dumps(
                    {
                        "iteration": iteration,
                        "seconds": iteration_record["seconds"],
                        "elapsed": time.perf_counter() - total_start,
                        "baseline": records[0],
                        "winner": records[selected],
                        "best_score": best_score,
                    }
                ),
                flush=True,
            )
            mean = 0.4 * mean + 0.6 * candidates[elite].mean(axis=0)
            deviation = np.maximum(0.4 * deviation + 0.6 * candidates[elite].std(axis=0), floor)
        # Re-evaluate finalists on separate worlds, preserving the root's final test set.
        final = np.stack([baseline, best_gains, mean] + [best_gains] * (args.population - 3))
        results = evaluate(final, validation)
        history["validation"] = [
            {**result, "gains": gains.tolist()}
            for result, gains in zip(results[:3], final[:3], strict=True)
        ]
        output.with_suffix(".json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        print("VALIDATION", json.dumps(history["validation"]), flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
