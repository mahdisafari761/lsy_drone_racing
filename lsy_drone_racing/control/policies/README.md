# Racing policy provenance

The experimental `Level2Fast` actor starts from Yuming Li's publicly released
`lv2_fast_policy.ckpt` and its observation definition in `my_rl_controller.py`:

- Repository: https://github.com/Yuming-Lee24/lsy_drone_racing
- Revision: `4e03f86f28243b426061f433633de1fa9d5f95be` (`25ws-main`)
- License: MIT; the original notice is preserved in [LICENSE](LICENSE).

The public tensors were loaded using PyTorch's `weights_only=True` conversion
and exported as plain NumPy arrays. Runtime loading uses `allow_pickle=False`;
PyTorch is not required for inference. The 88 public input features include two
historical states and the training wrapper's two-frame previous-action delay.
The adapter uses the current signed gate sequence, including repeated gates.

`level2_fast.npz` currently contains the original warm-start actor. It is a
**development baseline, not an accepted 90% / four-second controller**. Other
policy arrays in this directory are experimental comparison baselines.

Local adaptation uses `scripts/train_level2_fast.py`, with the original Level 2
physics, randomizations, disturbances and sensor limits. Training can additionally
start at later gates to learn individual sections. Such curriculum successes do
not count as full-race success. `scripts/screen_level2_fast.py` always screens
full races from the normal platform start; final acceptance must use independent
seeds through `scripts/benchmark_level2.py` and the official evaluator.

The original `level2_racing.py` remains the selected controller until the new
actor meets the requested acceptance criteria.
