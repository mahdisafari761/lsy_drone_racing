"""Freeze source/model hashes and validate a controller on predeclared full-race seeds.

Example::

    pixi run python -m scripts.validate_lead_release --controller level2_lead.py \
        --seed-start 500000 --n-runs 500 --output logs/lead_v2/release.json

Choose the release candidate before running a previously unused final seed set.
Do not tune on the final outcomes or repeatedly select candidates on the same set.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import inspect
import json
import math
import platform
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import crazyflow  # noqa: F401 - configure SciPy array API before SciPy imports
import numpy as np
from scipy.stats import t

from lsy_drone_racing.utils import load_config, load_controller
from scripts.benchmark_level2 import ROOT, benchmark, wilson_interval


def file_record(path: Path) -> dict[str, str]:
    """Identify an artifact by its resolved path and exact file content."""
    resolved = path.resolve()
    return {"path": str(resolved), "sha256": hashlib.sha256(resolved.read_bytes()).hexdigest()}


def release_manifest(controller: str, settings: dict[str, Any] | None = None) -> dict[str, Any]:
    """Record inherited controller sources, policy arrays, configuration and runtime."""
    path = Path(controller)
    if not path.is_absolute():
        path = ROOT / "lsy_drone_racing/control" / path
    cls = load_controller(path)
    settings = settings or {}
    for name, value in settings.items():
        if name.startswith("_") or not hasattr(cls, name) or callable(getattr(cls, name)):
            raise ValueError(f"Unknown public controller setting: {name}")
        setattr(cls, name, value)
    sources = {}
    for base in inspect.getmro(cls):
        try:
            source = inspect.getsourcefile(base)
        except TypeError:  # Built-in object has no Python source file.
            continue
        if source is not None and Path(source).is_file():
            source_path = Path(source).resolve()
            if source_path.is_relative_to(ROOT):
                sources[str(source_path.relative_to(ROOT))] = file_record(source_path)
    artifacts = {}
    for name in ("POLICY_FILE", "GAINS_FILE"):
        filename = getattr(cls, name, None)
        if isinstance(filename, str):
            artifact = path.parent / "policies" / filename
            artifacts[name] = file_record(artifact)
    config_path = ROOT / "config/level2.toml"
    config = load_config(config_path).to_dict()
    # These fields select the evaluation, rather than its physical race conditions.
    config.pop("controller", None)
    config["env"].pop("seed", None)
    config["sim"].pop("render", None)
    physical_hash = hashlib.sha256(
        json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    versions = {}
    for name in (
        "lsy_drone_racing",
        "numpy",
        "scipy",
        "jax",
        "jaxlib",
        "crazyflow",
        "gymnasium",
        "mujoco",
    ):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not-installed"
    return {
        "controller": file_record(path),
        "controller_settings": settings,
        "controller_sources": sources,
        "model_artifacts": artifacts,
        "config": file_record(config_path),
        "race_conditions_sha256": physical_hash,
        "benchmark": file_record(ROOT / "scripts/benchmark_level2.py"),
        "pixi_lock": file_record(ROOT / "pixi.lock"),
        "python": sys.version,
        "platform": platform.platform(),
        "versions": versions,
    }


def acceptance(result: dict, target_time: float, target_success: float) -> dict[str, Any]:
    """Separate strict empirical acceptance from statistical confidence statements."""
    episodes = result["episodes"]
    successful = np.asarray([e["official_time_s"] for e in episodes if e["success"]])
    count = len(episodes)
    successes = len(successful)
    rate = successes / count if count else 0.0
    mean = float(successful.mean()) if successes else None
    one_sided = wilson_interval(successes, count, z=1.6448536269514722)
    two_sided = wilson_interval(successes, count)
    if successes >= 30 and float(successful.std(ddof=1)) > 0:
        standard_error = float(successful.std(ddof=1) / math.sqrt(successes))
        radius = float(t.ppf(0.975, successes - 1)) * standard_error
        time_interval = [mean - radius, mean + radius]
        upper = mean + float(t.ppf(0.95, successes - 1)) * standard_error
    else:
        time_interval, upper = [None, None], None
    gate_durations = []
    for episode in episodes:
        if episode["success"]:
            split = np.asarray(episode["gate_pass_times_s"], dtype=float)
            gate_durations.append(np.diff(np.r_[0.0, split]))
    return {
        "target_mean_time_s": target_time,
        "target_success_rate_strictly_greater_than": target_success,
        "full_races": count,
        "successes": successes,
        "success_rate": rate,
        "successful_mean_time_s": mean,
        "strict_empirical_target_met": bool(
            mean is not None and mean <= target_time and rate > target_success
        ),
        "success_rate_wilson_95_two_sided": two_sided,
        "success_rate_wilson_95_one_sided_lower": one_sided[0],
        "successful_mean_time_95_two_sided_interval_s": time_interval,
        "time_interval_method": "Student-t approximation; requires at least 30 completions "
        "and nonzero sample variance because official times are quantized",
        "successful_mean_time_95_one_sided_upper_s": upper,
        "success_lower_confidence_bound_exceeds_target": one_sided[0] > target_success,
        "time_upper_confidence_bound_meets_target": upper is not None and upper <= target_time,
        "failure_gate_counts": dict(
            sorted(Counter(str(e["gates_passed"]) for e in episodes if not e["success"]).items())
        ),
        "mean_successful_gate_leg_s": np.mean(gate_durations, axis=0).tolist()
        if gate_durations
        else [],
        "interpretation": "Confidence describes the frozen controller and sampled distribution; "
        "successful-race time is conditional on completion, as in the official evaluator.",
    }


def main():
    """Freeze a candidate before a single declared acceptance run."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controller", default="level2_lead.py")
    parser.add_argument("--controller-settings", type=json.loads)
    parser.add_argument("--seed-start", type=int, default=500000)
    parser.add_argument("--n-runs", type=int, default=500)
    parser.add_argument("--target-time", type=float, default=3.30)
    parser.add_argument("--target-success", type=float, default=0.8)
    parser.add_argument("--output", type=Path, default=Path("logs/lead_v2/release.json"))
    parser.add_argument("--freeze-only", action="store_true")
    parser.add_argument("--require-confidence", action="store_true")
    args = parser.parse_args()
    if args.n_runs < 1 or args.seed_start < 0 or args.seed_start + args.n_runs > 2**32:
        raise ValueError("Invalid predeclared seed range")
    manifest = release_manifest(args.controller, args.controller_settings)
    manifest["predeclared_seeds"] = list(range(args.seed_start, args.seed_start + args.n_runs))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frozen_path = args.output.with_suffix(".frozen.json")
    if frozen_path.exists() or (not args.freeze_only and args.output.exists()):
        raise FileExistsError("Choose a new output path; release evidence is never overwritten")
    frozen_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if args.freeze_only:
        print(json.dumps({"frozen_manifest": str(frozen_path)}, indent=2))
        return
    result = benchmark(
        controller=args.controller,
        n_runs=args.n_runs,
        seed_start=args.seed_start,
        output=args.output,
        controller_settings=args.controller_settings,
        target_time=args.target_time,
        target_success=args.target_success,
    )
    after = release_manifest(args.controller, args.controller_settings)
    before_sources = {key: value for key, value in manifest.items() if key != "predeclared_seeds"}
    if after != before_sources:
        raise RuntimeError(
            "Release sources, model, environment or settings changed during evaluation"
        )
    report = acceptance(result, args.target_time, args.target_success)
    result["release_manifest"] = manifest
    result["acceptance"] = report
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(report, indent=2, allow_nan=False))
    decision = report["strict_empirical_target_met"]
    if args.require_confidence:
        decision = decision and report["success_lower_confidence_bound_exceeds_target"]
        decision = decision and report["time_upper_confidence_bound_meets_target"]
    if not decision:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
