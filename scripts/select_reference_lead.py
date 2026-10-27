"""Freeze a lead calibration using a separate development screen before final testing."""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from scripts.search_reference_policy import ROOT, build_rollout

paths = sorted(Path("logs").glob("reference-lead-*.npz"))
identity = np.tile([1.0, 0.0, 0.0, 1.0], (5, 1))
candidates = [("identity", identity)] + [
    (str(p), np.load(p, allow_pickle=False)["gains"]) for p in paths
]
env, rollout = build_rollout(4, 64, lead_control=True)
results = []
for start in range(0, len(candidates), 4):
    group = candidates[start : start + 4]
    padded = group + [("padding", identity)] * (4 - len(group))
    completed, times, passed = map(
        np.asarray,
        jax.device_get(
            rollout(jnp.asarray(np.stack([g for _, g in padded]), jnp.float32), jnp.asarray(2991))
        ),
    )
    for index, (name, gain) in enumerate(group):
        result = {
            "name": name,
            "success": float(completed[index].mean()),
            "mean": float(times[index][completed[index]].mean()),
            "gains": gain.tolist(),
        }
        results.append(result)
        print(json.dumps(result), flush=True)
    Path("logs/reference-lead-selection.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
feasible = [r for r in results if r["success"] >= 0.85]
selected = min(feasible or results, key=lambda r: r["mean"] + 20 * max(0, 0.85 - r["success"]))
destination = ROOT / "lsy_drone_racing/control/policies/level2_lead_gains.npz"
np.savez(destination, gains=np.asarray(selected["gains"], np.float32))
Path("logs/reference-lead-frozen.json").write_text(json.dumps(selected, indent=2), encoding="utf-8")
print("FROZEN", json.dumps(selected), flush=True)
env.close()
