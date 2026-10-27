"""Locate public last-safe states relative to gates and poles in failed development races."""

import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

report = json.loads(Path("logs/lead-failure-traces.json").read_text())
for episode in report["episodes"]:
    trace = episode["trace"]
    valid = [x for x in trace if not np.allclose(x["pos"], [-1,-1,-1])]
    sample = valid[-1]
    position, velocity = np.asarray(sample["pos"]),np.asarray(sample["vel"])
    predicted = position+.02*velocity
    gates = np.asarray(sample["gates_pos"])
    gate_rotation = Rotation.from_quat(sample["gates_quat"]).as_matrix()
    local = np.einsum("gji,gj->gi",gate_rotation,predicted[None]-gates)
    poles = np.asarray(sample["obstacles_pos"])
    pole_distances = np.linalg.norm(predicted[:2]-poles[:,:2],axis=-1)
    nearest_gate = int(np.argmin(np.linalg.norm(local,axis=-1)))
    nearest_pole = int(np.argmin(pole_distances))
    print(json.dumps({"seed":episode["seed"],"passed":episode["gates_passed"],
        "time":episode["official_time_s"],"position":position.tolist(),
        "velocity":velocity.tolist(),"nearest_gate":nearest_gate,
        "gate_local":local[nearest_gate].tolist(),
        "nearest_pole":nearest_pole,"pole_distance":float(pole_distances[nearest_pole]),
        "rpy":Rotation.from_quat(sample["quat"]).as_euler("xyz").tolist()}))
