# Phase D training, composition, and transfer comparison

Baseline: `105cd0c6b153f0e80a2593e43b7ffec7cc1f5e33`.

This table distinguishes behavior retained from the original repository from
the minimum changes required by `PROJECT_PROTOCOL.md`. It is audit evidence,
not a second scientific specification.

| Area | Preserved behavior | Necessary Phase D difference |
|---|---|---|
| RNN | One mRNN region, Softplus activation, 256 hidden units, `dt=10 ms`, `tau=20 ms`, recurrent noise `0.1`, input noise `0.01`, sigmoid output to six muscles. | The call into pinned `mRNNTorch@ac0c4f5` follows its current `(x0, input, h0=...)` signature. The baseline argument order raises a deterministic dimension error with that exact submodule. |
| Base loss | Full-trial position L1 plus the original rate, weight, muscle-activation, and simple-dynamics regularizers with unchanged weights. | None. No movement mask, velocity loss, endpoint weight, or digit-level reweighting is added. |
| Optimizer | Adam at `0.001` with gradient norm clipping at `1.0`, 75,000 updates, and validation every 500 updates. | The final continuation checkpoint is saved in addition to the best validation checkpoint; both contain optimizer state. |
| Batch sampling | One task, one speed, and one delay per batch; each sample independently draws one of eight spatial directions. | Tasks are digits. Full10 samples ten digits uniformly; heldout5 samples the other nine uniformly and never activates rule column 5. |
| Validation | All trained tasks, 32 directions, ten validation speeds, and mean full-trial position L1 as the sole checkpoint criterion. | Digit-dependent movement duration comes from the Phase B geometry. A fixed validation RNG state makes repeated checkpoint comparisons reproducible without changing conditions or loss. |
| Base-model separation | Each call to `train.py::train_subsets_base_model` creates a fresh policy and optimizer and uses the original rollout, five-term loss, update, and `do_eval` loops. | Thin JSON adapters pass either all ten digit environments or the nine non-5 environments. Full10 and heldout5 use different configurations, directories, checkpoints, and runner logs; neither base entry accepts a source checkpoint. |
| Composition | The original `composite_input_optimization` core is retained: Adam, learning rate `0.1`, 250 iterations, no network noise, and full-trial position L1. | A target digit uses ten rule coefficients with its own coefficient fixed to zero. The other nine coefficients start at one and remain independent across the eight validation directions. Only external coefficients are trainable; the full10 network is bitwise frozen. |
| Composition conditions | Batch size 8, validation-direction indices `[0,4,8,12,16,20,24,28]`, validation speed index 9, and custom delay 150. | The formal entry runs all ten target digits rather than selecting a favorable subset. |
| Transfer | The original held-out training entry supplies Adam at `0.001`, gradient clipping at `1.0`, full-trial position L1 only, and checkpoint selection on held-out behavior. | The entry now loads the independent heldout5 checkpoint, explicitly reinitializes only input-weight column 5, and applies a strict gradient mask. Every other parameter and rule column is audited bitwise unchanged. |
| Run provenance | Checkpoints include model and optimizer state. | Best and final continuation checkpoints also record the update, resolved protocol configuration, variant, and Python/NumPy/Torch RNG state. Each runner records Git and submodule heads, original baseline, config and hash, seed, CPU environment, full log, source-checkpoint hash where applicable, output hashes, and an archive hash. The formal Python process suppresses bytecode writes and rechecks both repositories after the run so importing the pinned submodule cannot dirty it between sequential experiments. |

Phase D implementation and tests do not authorize or start a development seed,
the 75,000-update runs, composition optimization, or transfer training. Every
experiment runner stops unless Phase E user authorization is supplied
explicitly.

## Pinned mRNNTorch compatibility mapping

The patch is limited to the call boundary in `RNNPolicy.forward`:

| Existing wrapper value | Pinned `mRNN.forward` parameter | Change in value or meaning |
|---|---|---|
| `x` | positional `x0` | None |
| `obs[:, None, :]` | positional `inp` | None; the singleton time dimension is unchanged |
| `h` | keyword `h0` | None |
| `*args` | remaining positional arguments | None |
| `noise` | keyword `noise` | None |
| returned `(x, h)` | returned `(x, h)` | None; the existing singleton time squeeze is retained |

No submodule source, parameter tensor, state-dict key, initialization rule,
noise magnitude, or optimizer setting is changed by this compatibility patch.
