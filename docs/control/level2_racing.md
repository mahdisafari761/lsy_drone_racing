# Level 2 racing controller

`level2_racing.py` is selected in `config/level2.toml`. The attitude interface,
randomizations, disturbances, sensor range, track geometry and gate order are
unchanged. The original `level2_mpc.py` remains available.

**The requested combination of approximately 90% success and a four-second mean
lap has not been achieved.** The implemented controller improves robustness and
reduces measured lap times to approximately 14 seconds.

## Design

The controller inherits the observation-driven visibility planner and the
20-stage, 1.2-second acceleration MPC from `Level2MPC`. It changes:

- Tracking weights, acceleration limits and waypoint reference speed to reduce
  the baseline's slow convergence to each waypoint.
- Gate entry logic to permit a rolling crossing when both lateral position and
  predicted lateral crossing error are below 5 cm. Otherwise it aligns first.
- Aperture targeting: if the centerline is obstructed, small lateral or vertical
  offsets use another part of the same opening. Gate direction and repeated
  physical gates still follow the public signed gate sequence.
- Gate exits: available space behind the gate determines when to begin the next
  approach, rather than requiring convergence to a fixed exit waypoint.
- Acceleration filtering: local second-order distance constraints brake closing
  motion near observed poles and oriented gate collision boxes. Nominal objects
  retain larger uncertainty margins until they are revealed by the sensor.

The filter uses a finite number of projections and bounded acceleration. It is
not a formal collision guarantee, particularly when constraints conflict, an
object is revealed inside the stopping distance, or attitude/thrust saturation
changes the commanded acceleration. Parameters target the configured CF21B_500
vehicle. Real-drone performance has not been validated.

## Seeded validation, 2026-10-21

Each episode is reset with its own seed. This gives comparable randomized tracks
across controllers regardless of their different episode lengths. All five
signed gate entries must be passed for success. Mean times include successful
episodes only and count every simulated control step, including the final one.

| Test set | Controller | Successful runs | Mean successful lap |
| --- | --- | ---: | ---: |
| Development: seeds 1000–1019 | Original Level2MPC | 14/20 (70%) | 23.17 s |
| Development: seeds 1000–1019 | Selected Level2Racing | 19/20 (95%) | 14.42 s |
| Separate validation: seeds 20000–20019 | Selected Level2Racing | 19/20 (95%) | 14.39 s |
| Official `scripts/evaluate.py`, random seeds | Selected Level2Racing | 18/20 (90%) | 14.86 s |

The separate validation set was not used for tuning the selected parameters.
Both unsuccessful selected-controller episodes in the seeded development and
separate validation sets timed out. The official evaluation had one timeout and
one early termination. Finite 4D commands
were checked throughout the simulations. The 95% observed rate is a sample
result, not a guaranteed rate for future random seeds.

The separate validation's 99th-percentile control computation time was 8.41 ms;
four individual calls exceeded the 20 ms control period. Other simulations were
running concurrently, so this is not an isolated real-time performance test.

A shorter-horizon alternative and continuous quintic trajectory candidates were
also explored. They did not establish the requested four-second/90% combination;
the selected implementation retains the more reliable gate-aligned planner.

## Reproduce

Run these commands in the project environment:

```bash
python scripts/sim.py --config level2.toml
python scripts/evaluate.py
python scripts/benchmark_level2.py --controller level2_racing.py \
    --seed 20000 --runs 20 --output /tmp/level2-racing.json
python scripts/benchmark_level2.py --controller level2_mpc.py \
    --seed 1000 --runs 20 --output /tmp/level2-baseline.json
python -m unittest discover -s tests/unit/control -v
ruff check .
ruff format --check .
```

The benchmark records every episode, configuration/controller hashes, gate
crossing times, success rate, successful mean lap time and controller latency.
The official evaluation script reports its loop's zero-based time, 0.02 seconds
less than the benchmark for an otherwise identical 50 Hz episode.

The added regression tests exercise braking with a delayed acceleration response,
use of a clear aperture offset, and rotation of frame constraints with the gate.
