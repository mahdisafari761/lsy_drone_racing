# Level 2 Sprint

The user target is mean successful official race time <=3.327 s and success rate
>=75%. Individual successful races can take longer than this average.

`level2_sprint.py` uses the original reference actor and the adapted 1,024,000-step
actor. Only the first section uses a distance-dependent mixture: the adapted
actor dominates far from the public target, transitioning smoothly to the
original actor between 1.6 m and 0.6 m. Later sections use the original actor.
Both actors share one Lead filter and one previous-action history.

The portable `policies/level2_sprint.npz` contains both complete networks,
normalization and the frozen 5x4 calibration from development candidate 004-15.
It is used for both POLICY_FILE and GAINS_FILE. Runtime needs no `logs/` files,
training software, JAX policy inference or preview simulator. The existing base
controller uses a nominal simulator only to obtain the ordinary action bounds.
MIT attribution remains in `policies/level2_reference_LICENSE`.

## Measurements

| Evaluation | Full races | Success | Mean successful official time |
| --- | ---: | ---: | ---: |
| Selection/development, seeds 43000-43199 | 200 | 158/200 (79%) | 3.326329 s |
| Frozen independent acceptance, seeds 500000-500499 | 500 | 376/500 (75.2%) | 3.321862 s |

The independent acceptance test meets both empirical targets. The policy,
inherited controller sources, physical configuration, runtime versions and Pixi
lock were fingerprinted before evaluation and verified unchanged afterwards.
The policy SHA-256 is
`99c830cdf5d5e8f208f5a8627ad5bf28bcc6eb92977f2104efbb94eec3108e39`.

The reliability margin is small. The two-sided 95% Wilson interval for the
independent success rate is 71.23%-78.78%; the corresponding successful-mean-time
interval is 3.30497-3.33876 s. These are measured point estimates, not a guarantee
for every future 20-race batch or a confidence-level proof of both thresholds.
The final seed set was used once for this frozen candidate and was not used for
parameter search. The development and independent samples are reported separately.

The physical Level 2 configuration is unchanged. Time follows `scripts/sim.py`,
`(steps - 1) / 50`, and the mean excludes unsuccessful races. Complete JSON
records, confidence intervals, timing measurements and source/model fingerprints
are stored under `logs/lead_v5/` locally.

Packaging verification also found all 22 archive arrays byte-identical to their
selected sources and actions/history bit-identical in 70 synthetic states,
including section transitions, signed gate directions and reset.

Across the independent races, control plus callback time averaged 2.16 ms,
with p95 3.14 ms and maximum 10.55 ms. There were no exceedances of the original
20 ms control budget.

## Technical validation

- Repository-wide Ruff check: passed.
- Ruff format check: all 151 Python files formatted.
- Complete project suite in the locked Pixi `tests` environment: **130 passed,
  22 skipped, 52 subtests passed**, in 1031.89 s. The 25 JAX cast warnings are the
  same warning category present in the previous baseline suite.
- The test launcher places temporary files inside `logs/lead_v5/test-temporary`
  and disables pytest's result cache to accommodate Windows sandbox permissions.
  Test collection remains the complete `tests` directory.

## Run

The Level 2 configuration selects this controller in attitude mode:

```powershell
.\.tools\pixi\pixi.exe run --as-is python .\scripts\sim.py --config level2.toml --n_runs 20 --render True
```

For a headless 20-race competition evaluation and summary:

```powershell
.\.tools\pixi\pixi.exe run --as-is python .\scripts\evaluate.py
```

Required additions to the existing repository are `level2_sprint.py`,
`level2_distance_actor.py`, `policies/level2_sprint.npz`, this document and the
controller selection in `config/level2.toml`. The policy metadata JSON records
provenance; it is not opened during a race. Existing `level2_phase_actor.py`,
`level2_lead.py`, `level2_reference.py`, `controller.py` and the license are also
required as ordinary dependencies. Search scripts are development tools.
