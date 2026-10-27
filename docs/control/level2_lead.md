# Level 2 racing controller: measured status

The selected controller is `lsy_drone_racing/control/level2_lead.py`, using the
attitude interface. It improves the initial controller substantially, but the
requested mean time <=3.2 seconds and success rate >=80% are not yet jointly met.

## Measurements on 27 October 2026

| Evaluation | Full starts | Success | Mean successful time |
| --- | ---: | ---: | ---: |
| Initial controller, original official evaluator | 20 | 70% | 4.8529 s |
| New frozen controller, independent seeds 10000-10099 | 100 | 81% | 3.3158 s |
| New controller, separate original official evaluator | 20 | 85% | 3.2565 s |

The 100-race median is 3.32 s, p95 is 3.56 s, and the maximum successful time
is 3.68 s. The binomial 95% Wilson interval for success is 72.2%-87.5%; an observed
81% is a sample estimate, not a guarantee that the population success rate is >=80%.
The independent mean misses the requested time by 0.1158 s.

The complete independent record, including seed list, file hashes, failures,
gate splits and control timing, is `logs/level2-lead-holdout-100.json`.
The official script writes its separate result to `evaluation.csv`.
Logs and experimental checkpoints are local files excluded by the existing ignore rules.

Mean control-plus-callback time was 0.415 ms, p95 0.475 ms, maximum 1.909 ms;
none of the measured loops exceeded the configured 20 ms control budget.
Reported race times follow the unmodified official zero-based step convention.

## Controller design

The warm start is Xiaolong-Li1904's MIT-licensed public NumPy racing actor at
revision `1cca34f2476d7791bdb7074fc9cf3dc32b4fbc27`. Its notice is preserved
and packaged with the policy. Source and adapter details are in
`lsy_drone_racing/control/policies/README.md`.

The observation encoder uses heading-frame velocity, body rotation and rates,
the current and following signed gate entries, detection flags, the two nearest
poles and the previous action. A four-layer ELU network returns bounded attitude
and thrust. This avoids a PyTorch runtime or model compilation during controller
initialization.

The new controller adds frozen parameters for each of the five race sections:
velocity-feature calibration, a small normalized action lead filter, collective
thrust bias, and gate-translation feature calibration. The lead filter anticipates
part of the attitude servo lag while retaining the original action bounds.
Crossing all five configured gate entries ends the controller immediately.

Calibration used complete original Level 2 races in vectorized development
simulations, then a separate development screen. `logs/reference-lead-frozen.json`
records the selected gains before the independent 100-race evaluation.
The final test outcomes were not used to retune the selected gains.

The controller receives only the public `obs`, `info` and configuration. Physics,
randomization, disturbances, sensor range, gate sequence, bounds, termination,
and the official evaluator remain unchanged. The only race-config change is
`controller.file = "level2_lead.py"`; `env.control_mode` remains `attitude`.

## Reproduce with the local Pixi installation

```powershell
.\.tools\pixi\pixi.exe run --as-is evaluate
.\.tools\pixi\pixi.exe run --as-is python scripts/benchmark_level2.py --controller level2_lead.py --n-runs 100 --seed-start 10000 --output logs/reproduce-lead.json
.\.tools\pixi\pixi.exe run --as-is python -m unittest discover -s tests/unit -p test_reference_policy.py -v
.\.tools\pixi\pixi.exe run --as-is python -m unittest discover -s tests/unit -p test_benchmark_level2.py -v
```

Replaying these seeds reproduces an evaluation, not a new independent test.
Use a fresh predeclared seed set for evaluating future changes.

`scripts/search_reference_policy.py` and `scripts/select_reference_lead.py`
retain the calibration workflow. Unsuccessful PPO, body-rate, spline, impulse
adaptation and other prototypes/checkpoints are archived under `logs/experiments/`.
Their development results are not claimed as final acceptance evidence.
