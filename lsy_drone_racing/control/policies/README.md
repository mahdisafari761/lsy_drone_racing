# Racing policy provenance

## Selected Level2Lead policy

`level2_reference.npz` is Xiaolong-Li1904's public, MIT-licensed Level 2 actor:

- Repository: https://github.com/Xiaolong-Li1904/lsy_drone_racing
- Revision: `1cca34f2476d7791bdb7074fc9cf3dc32b4fbc27`
- Source archive: `lsy_drone_racing/control/level2_rl_policy.npz`
- Source definitions: `rl/features.py`, `rl/actions.py`, `rl/policy_np.py`,
  and `control/level2_rl_controller.py`.
- Preserved license: [level2_reference_LICENSE](level2_reference_LICENSE).

The actor uses 46 public observation features and the previous normalized action.
It supports the current five-entry signed gate sequence, including repeated gates.
Runtime inference is NumPy only and loads arrays with `allow_pickle=False`.

`level2_lead_gains.npz` contains the locally tuned five-by-four calibration:
heading-frame velocity scale, normalized action lead gain, collective-thrust bias,
and gate-translation feature scale. Parameters are fixed before a race; neither
the model nor the gains read hidden simulation data or race random seeds.

The selected controller was measured at **3.3158 s and 81/100 successes** on
independent full-race seeds 10000 through 10099. A separate official evaluator
run gave **3.2565 s and 17/20 successes**. The requested <=3.2 s / >=80% target
is **not yet fully met**. See [controller report](../../../docs/control/level2_lead.md).

## Legacy Level2Fast policy

The earlier `Level2Fast` actor starts from Yuming Li's publicly released
`lv2_fast_policy.ckpt` and its observation definition in `my_rl_controller.py`:

- Repository: https://github.com/Yuming-Lee24/lsy_drone_racing
- Revision: `4e03f86f28243b426061f433633de1fa9d5f95be` (`25ws-main`)
- License: MIT; the original notice is preserved in [LICENSE](LICENSE).

The public tensors were loaded using PyTorch's `weights_only=True` conversion
and exported as plain NumPy arrays. Runtime loading uses `allow_pickle=False`;
PyTorch is not required for inference. The 88 public input features include two
historical states and the training wrapper's two-frame previous-action delay.
The adapter uses the current signed gate sequence, including repeated gates.

`level2_predictive_85.npz` contains the earlier actor with its existing local
residual. Its initial official evaluation gave 4.8529 s and 14/20 successes.
It remains available for comparison through `level2_predictive_85.py`.
