"""Screen gate-center target corrections on original full-race development starts."""

import json
from pathlib import Path

from scripts.benchmark_level2 import benchmark

for gain in [1.05,1.1,1.2]:
    result=benchmark(controller="level2_guarded_lead.py",n_runs=100,seed_start=30000,
        controller_settings={"STRENGTH":0., "BLIND_SPEED_GAIN":gain},
        output=Path(f"logs/blind-speed-lead-{gain}.json"),target_time=3.3)
    print(json.dumps({"gain":gain,"summary":result["summary"]}),flush=True)
