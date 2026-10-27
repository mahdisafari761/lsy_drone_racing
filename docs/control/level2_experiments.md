# Level 2 controller experiments

## State-interface and blind-approach experiments, 27 October 2026

Development screening used 20 complete races with reset seeds 100 through 119,
the original randomized Level 2 track and disturbances, and the normal platform
start. These are tuning results, not independent acceptance results. Times below
average successful runs only and follow the official evaluator's zero-based tick
convention.

| Experimental variant | Success | Mean successful time |
| --- | ---: | ---: |
| Predictive actor, state adapter with existing inversion clamps | 45% | 5.387 s |
| State adapter with precise heading/thrust inversion | 75% | 5.279 s |
| Precise inversion, rotation-error body-rate gain -0.25 | 75% | 5.177 s |
| Precise inversion, body-rate gain +0.25 | 50% | 5.264 s |
| Precise inversion, body-rate gain +0.5 | 40% | 5.330 s |
| AGILE predictor with its first-gate candidate body rates | 35% | 5.326 s |
| Attitude predictor, unseen speed 2.2 m/s, margin 0.015 m, horizon 25 | 60% | 5.085 s |
| Attitude predictor, unseen speed 1.8 m/s, same margin/horizon | 55% | 5.080 s |

The state-interface adapter exactly inverted the public nominal Mellinger
position-to-attitude map and loaded its nominal parameters with
`load_fn_params(state2attitude, config.sim.drone)`. It used no hidden race state.
Reducing inversion clamps from 0.1 to 1e-5 for the desired vertical axis and from
0.2 to 0.01 for thrust-axis alignment improved this particular adapter. Positive
body-rate feedback then reduced success substantially. The pretrained actor
appears sensitive to changes in its attitude response.

The blind-approach variants limited predicted speed before the first exact gate
observation. Most failures still occurred before passing the first gate. Neither
experiment met 3.2 seconds and 80% success, and neither was selected. Their four
temporary controller/screening source files were removed after this record was
written. The shared runtime helpers and race configuration were not changed by
these experiments.
