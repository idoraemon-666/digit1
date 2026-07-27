# Phase E delivery review

## 1. Review status and boundary

Review target: server head
`b7b9b28189e33010a1ee5da28a0a27b0bd101a59`, whose direct parent is the
Phase D v2 implementation commit
`63b7bcbf913816a44c027a6e14a0c4d6f92363bd`.

Scientific authority: `PROJECT_PROTOCOL.md`.

This document performs the delivery review required by protocol lines
1077–1098. It does not authorize or start a development seed, either base
training run, composition optimization, or digit-5 transfer. The current
evidence supports **ready for user review, with no implementation blocker
found**. That status is not an experiment authorization.

The authoritative runtime evidence comes from the AutoDL CPU repository and
downloaded result archives. The local Git worktree is an assembled review
worktree and is intentionally not used as the authoritative server HEAD.

## 2. Identity and provenance

| Item | Frozen value / evidence | Result |
|---|---|---|
| Original repository baseline | `105cd0c6b153f0e80a2593e43b7ffec7cc1f5e33` | Verified throughout Phase A–D |
| Current server HEAD | `b7b9b28189e33010a1ee5da28a0a27b0bd101a59` | Verified |
| Phase D implementation parent | `63b7bcbf913816a44c027a6e14a0c4d6f92363bd` | Verified |
| Pinned submodule | `ac0c4f589eae37bbde63968912925de99232e306` | Unchanged |
| Baseline device | CPU | Verified |
| Python / PyTorch / MotorNet / NumPy | 3.10.20 / 2.6.0+cpu / 0.2.0 / 2.2.6 | Recorded in server archives |
| Phase D main result SHA-256 | `61b2f9f0f8ac81465954e76e54684c2f404178cfa94bae50f0efd17b25cfa0bb` | Outer and inner hashes verified locally |
| Runner-cleanliness result SHA-256 | `449810f67e309826c0e4fba15c69ec76bea85f31fe5ac42f8cc2b4d0f1c1cded` | Outer and inner hashes verified locally |

Result directories:

- `artifacts/server-results/digit-writing-phaseD-v2-implementation-63b7bcb-run1`
- `artifacts/server-results/digit-writing-phaseD-v2.1-runner-cleanliness-b7b9b28-run1`

## 3. Protocol-mandated delivery checklist

| # | Required delivery | Evidence | Finding |
|---:|---|---|---|
| 1 | Modified-file list | Section 4 and both Phase D commit manifests | Complete |
| 2 | Itemized consistency with `105cd0c...` | `PHASE_C_ENVIRONMENT_BASELINE_DIFF.md`, `PHASE_D_TRAINING_BASELINE_DIFF.md`, Section 5 | Complete |
| 3 | Every necessary difference and reason | Section 5, including pinned mRNN signature mapping | Complete |
| 4 | Ten prescribed-direction trajectory figures | Phase B figure manifest and Section 6.1 | 10/10, hashes verified |
| 5 | Eight direction-specific `2×5` figures | Phase B figure manifest and Section 6.2 | 8/8, hashes verified |
| 6 | Geometry/time/workspace audit report | `DIGIT_GEOMETRY_PROTOCOL_AUDIT.md` and Section 7 | Complete; zero blockers |
| 7 | All tests | Current 51-test logs and Section 8 | 51/51, zero skipped |
| 8 | Two base models and composition/transfer entries | Four JSON configs, four wrappers, Section 9 | Complete and separated |
| 9 | Updated handoff | `NEXT_SESSION_HANDOFF.md` | Complete |
| 10 | Declaration that seed-42 training has not begun | Both Phase D execution scopes and Section 11 | Verified zero runs |

## 4. Modified-file inventory

### 4.1 Root scientific and runtime files

- `PROJECT_PROTOCOL.md`: sole project authority, including the accepted digit-6
  correction and phase stop lines.
- `README.md`, `NEXT_SESSION_HANDOFF.md`: project state and continuation
  boundary.
- `config.py`: thin dispatch for the four protocol run kinds.
- `envs.py`: shared digit environment preserving the original 28-dimensional
  observation and MotorNet closed loop.
- `model.py`: pinned mRNNTorch call adaptation and digit-5 rule-column
  isolation.
- `train.py`: digit environment mapping, thin base adapters, checkpoint
  provenance, deterministic validation wrapper, and isolated transfer entry.
- `PHASE_C_ENVIRONMENT_BASELINE_DIFF.md`: environment consistency evidence.
- `PHASE_D_TRAINING_BASELINE_DIFF.md`: training, composition, transfer, and
  compatibility consistency evidence.
- `PHASE_E_DELIVERY_REVIEW.md`: this review.

`losses.py` and `utils.py` remain unchanged from the original baseline. The
server-side original-core diff contains changes only in `config.py`, `envs.py`,
`model.py`, and `train.py`.

### 4.2 Protocol configurations

- `configurations/digit_original_protocol_geometry.json`
- `configurations/digit_original_protocol_full10_dev42.json`
- `configurations/digit_original_protocol_heldout5_dev42.json`
- `configurations/digit_original_protocol_composition.json`
- `configurations/digit_original_protocol_transfer5.json`

### 4.3 Digit-specific implementation

- `digit_writing/__init__.py`
- `digit_writing/geometry.py`
- `digit_writing/geometry_audit.py`
- `digit_writing/protocol_audit.py`
- `digit_writing/experiments.py`

### 4.4 Tests

- `tests/test_digit_geometry.py`
- `tests/test_digit_environment.py`
- `tests/test_geometry_audit.py`
- `tests/test_protocol_audit.py`
- `tests/test_phase_d_protocol.py`

### 4.5 Server entries and audit runners

- `server/run_digit_original_protocol_geometry_audit.sh`
- `server/run_digit_geometry_protocol_audit.sh`
- `server/run_digit_original_protocol_experiment.sh`
- `server/run_digit_original_protocol_full10_dev42.sh`
- `server/run_digit_original_protocol_heldout5_dev42.sh`
- `server/run_digit_original_protocol_composition.sh`
- `server/run_digit_original_protocol_transfer5.sh`

### 4.6 Project control and reproducibility support

- `.gitattributes`, `.gitignore`, `AGENTS.md`
- `AUTODL_SERVER_USAGE_GUIDE.md`
- `server/README.md`
- `server/create_compat_environment.sh`
- `server/environment-linux-cpu.yml`
- `server/environment-linux-cu118.yml` (retained as non-baseline support only)
- `server/requirements-linux-common.txt`
- `server/requirements-linux-cpu.txt`
- `server/requirements-linux-cu118.txt` (retained as non-baseline support only)

The original `analysis/`, full `experiments/`, `config.txt`,
`configurations/mrnn.json`, and unrelated original analysis configurations are
intentionally absent from the active tree but remain byte-exact in Git history
at `105cd0c...`. This deletion keeps the active project scoped; the only
original experiment code migrated into runtime is the explicitly authorized
composition core. No code or result was imported from the historical
digit-writing project.

## 5. Baseline consistency and necessary differences

| Area | Preserved contract | Necessary protocol difference / reason | Review |
|---|---|---|---|
| Git and dependency identity | Direct ancestry from `105cd0c...`; pinned `mRNNTorch@ac0c4f5` | None | Pass |
| Motor plant and output | `RigidTendonArm26`, `MujocoHillMuscle`, six sigmoid muscle outputs, CPU | None | Pass |
| Observation | Original 28 columns and cue timing | First ten rule columns now denote digits 0–9 | Pass |
| Trial sampling | One task, speed, and delay per batch; per-sample spatial direction | Digit task and arc-length-dependent movement duration | Pass |
| Base loss | Full-trial position L1 plus rate, weight, muscle, and simple-dynamics terms | None | Pass |
| Base optimization | Adam `0.001`, batch 32, clip 1, 75,000 updates, validation every 500 | Seed/config mapping and best/final continuation provenance only | Pass |
| Validation | All trained tasks, 32 directions, ten speeds, mean position L1 selection | Fixed validation RNG for reproducibility without changing conditions | Pass |
| full10 / heldout5 | Same original base rollout, loss, update, and `do_eval` core | Independent 10- and 9-digit environment sets, models, optimizers, directories, and checkpoints | Pass |
| RNN forward | Existing `x`, observation, `h`, and noise values | Reordered only at the pinned API boundary to `(x0, inp, h0=...)`; baseline order fails dimensionally | Pass |
| Composition | Original Adam `0.1`, 250 iterations, no network noise, trajectory L1 | All nine non-target external coefficients are free per direction; target coefficient fixed at zero; full10 network frozen | Pass |
| Transfer | Original transfer optimizer, clip, sampling, L1, and held-out validation | Start only from heldout5; reinitialize and train input column 5 with strict mask; audit every other value bitwise | Pass |
| Checkpoints | Model and optimizer states | Add resolved config, variant, update, validation value, and RNG states for continuation/audit | Pass |
| Runner provenance | CPU and clean-repository preflight | Exact run-label authorization, source-checkpoint hash, bytecode suppression, post-run cleanliness, archive hashes | Pass |

The full compatibility mapping is in
`PHASE_D_TRAINING_BASELINE_DIFF.md`. No network layer, state-dict key, loss
weight, optimizer value, geometry value, or submodule source was changed by
the mRNN call adaptation.

## 6. Required figures

All files below are under:

`artifacts/server-results/digit-writing-phaseB-geometry-audit-e948eaa-run2/audit/figures/`

Their hashes were rechecked against `figure_manifest.json` during Phase E.

### 6.1 Ten prescribed-direction figures

| Digit | Figure |
|---:|---|
| 0 | [prescribed_digit_0.png](artifacts/server-results/digit-writing-phaseB-geometry-audit-e948eaa-run2/audit/figures/prescribed_digit_0.png) |
| 1 | [prescribed_digit_1.png](artifacts/server-results/digit-writing-phaseB-geometry-audit-e948eaa-run2/audit/figures/prescribed_digit_1.png) |
| 2 | [prescribed_digit_2.png](artifacts/server-results/digit-writing-phaseB-geometry-audit-e948eaa-run2/audit/figures/prescribed_digit_2.png) |
| 3 | [prescribed_digit_3.png](artifacts/server-results/digit-writing-phaseB-geometry-audit-e948eaa-run2/audit/figures/prescribed_digit_3.png) |
| 4 | [prescribed_digit_4.png](artifacts/server-results/digit-writing-phaseB-geometry-audit-e948eaa-run2/audit/figures/prescribed_digit_4.png) |
| 5 | [prescribed_digit_5.png](artifacts/server-results/digit-writing-phaseB-geometry-audit-e948eaa-run2/audit/figures/prescribed_digit_5.png) |
| 6 | [prescribed_digit_6.png](artifacts/server-results/digit-writing-phaseB-geometry-audit-e948eaa-run2/audit/figures/prescribed_digit_6.png) |
| 7 | [prescribed_digit_7.png](artifacts/server-results/digit-writing-phaseB-geometry-audit-e948eaa-run2/audit/figures/prescribed_digit_7.png) |
| 8 | [prescribed_digit_8.png](artifacts/server-results/digit-writing-phaseB-geometry-audit-e948eaa-run2/audit/figures/prescribed_digit_8.png) |
| 9 | [prescribed_digit_9.png](artifacts/server-results/digit-writing-phaseB-geometry-audit-e948eaa-run2/audit/figures/prescribed_digit_9.png) |

### 6.2 Eight training-direction `2×5` figures

| Direction | Figure |
|---:|---|
| 0° | [training_direction_0_000.0deg.png](artifacts/server-results/digit-writing-phaseB-geometry-audit-e948eaa-run2/audit/figures/training_direction_0_000.0deg.png) |
| 45° | [training_direction_1_045.0deg.png](artifacts/server-results/digit-writing-phaseB-geometry-audit-e948eaa-run2/audit/figures/training_direction_1_045.0deg.png) |
| 90° | [training_direction_2_090.0deg.png](artifacts/server-results/digit-writing-phaseB-geometry-audit-e948eaa-run2/audit/figures/training_direction_2_090.0deg.png) |
| 135° | [training_direction_3_135.0deg.png](artifacts/server-results/digit-writing-phaseB-geometry-audit-e948eaa-run2/audit/figures/training_direction_3_135.0deg.png) |
| 180° | [training_direction_4_180.0deg.png](artifacts/server-results/digit-writing-phaseB-geometry-audit-e948eaa-run2/audit/figures/training_direction_4_180.0deg.png) |
| 225° | [training_direction_5_225.0deg.png](artifacts/server-results/digit-writing-phaseB-geometry-audit-e948eaa-run2/audit/figures/training_direction_5_225.0deg.png) |
| 270° | [training_direction_6_270.0deg.png](artifacts/server-results/digit-writing-phaseB-geometry-audit-e948eaa-run2/audit/figures/training_direction_6_270.0deg.png) |
| 315° | [training_direction_7_315.0deg.png](artifacts/server-results/digit-writing-phaseB-geometry-audit-e948eaa-run2/audit/figures/training_direction_7_315.0deg.png) |

Manifest:
`artifacts/server-results/digit-writing-phaseB-geometry-audit-e948eaa-run2/audit/figure_manifest.json`.

## 7. Geometry, timing, workspace, and rollout audit

Primary report:

`artifacts/server-results/digit-writing-geometry-protocol-audit-e116b08-run1/audit/DIGIT_GEOMETRY_PROTOCOL_AUDIT.md`

Verified findings:

- zero blocking issues;
- all shared primitive, join-position, finite-value, and analytic tangency
  constraints passed;
- 3,440 MotorNet workspace conditions passed;
- 130 no-training rollout groups covering 3,440 direction conditions passed;
- all rollout observations retained the 28-dimensional contract and finite
  diagnostic loss inputs;
- no checkpoint, optimizer, development seed, or training was used.

Known non-blocking risk: digit movement duration has a longest/shortest ratio
of `6.318181818`, compared with `2.0` for the original tasks. Digits 1 and 7
are short. This review does not convert that observation into a new failure
threshold and does not change geometry, speed, or timing.

## 8. Test evidence

Current suite composition:

| Test file | Methods |
|---|---:|
| `test_digit_geometry.py` | 25 |
| `test_digit_environment.py` | 5 |
| `test_geometry_audit.py` | 3 |
| `test_protocol_audit.py` | 5 |
| `test_phase_d_protocol.py` | 13 |
| **Total** | **51** |

Authoritative final log:

`artifacts/server-results/digit-writing-phaseD-v2.1-runner-cleanliness-b7b9b28-run1/tests.log`

Result: `Ran 51 tests`, `OK`, zero skipped, Python and tee exit codes zero.
The test process used a temporary bytecode cache prefix; no submodule cache was
created. The result repository-status file is empty.

The earlier Phase D validation also passed 51/51 with zero skipped. Its script
stopped after the successful commit because generated submodule bytecode
tripped the post-commit cleanliness check. Only the two regenerable `.pyc`
files were removed, and the retained logs were used to resume archive creation
without rerunning tests. Commit `b7b9b28...` permanently removes that
operational risk for formal runners.

## 9. Experiment-entry separation

| Entry label | Config | Permitted source | Output | Current authorization |
|---|---|---|---|---|
| `full10-dev42` | `digit_original_protocol_full10_dev42.json` | None; fresh model and optimizer | `runs/digit_original_protocol/full10/dev42` | Not authorized |
| `heldout5-dev42` | `digit_original_protocol_heldout5_dev42.json` | None; fresh model and optimizer | `runs/digit_original_protocol/heldout5/dev42` | Not authorized |
| `composition-dev42` | `digit_original_protocol_composition.json` | full10 best checkpoint only | `runs/digit_original_protocol/composition/dev42` | Not authorized |
| `transfer5-dev42` | `digit_original_protocol_transfer5.json` | heldout5 best checkpoint only | `runs/digit_original_protocol/transfer5/dev42` | Not authorized |

The generic runner requires `DIGIT_PROTOCOL_AUTHORIZED_RUN` to equal the exact
label. Composition cannot run before full10 exists; transfer cannot run before
heldout5 exists. No base entry accepts a source checkpoint.

## 10. Review findings

### Blocking findings

None in the implementation, frozen geometry, test evidence, entry separation,
or audit provenance.

### Non-blocking risks and unknowns

1. The `6.318` digit duration spread is substantially larger than the original
   task spread of `2.0`.
2. All accepted evidence is CPU-specific. CUDA is outside this result lineage.
3. Behavioral learning quality, convergence, and final composition/transfer
   success remain unknown because no new-protocol training has been run.
4. The 75,000-update base runs are expensive; the current review verifies the
   executable protocol, not their eventual scientific outcome.

These risks must be interpreted from formal results; they do not authorize
post-hoc changes to geometry, speed, loss, or network size.

## 11. Zero-training declaration and approval boundary

The following independently archived markers are all zero:

```text
DEVELOPMENT_SEED_RUNS=0
BASE_TRAINING_RUNS=0
COMPOSITION_RUNS=0
TRANSFER_RUNS=0
RUNS_DIRECTORY_CREATED=0
```

Therefore seed-42 new-protocol training has **not** begun.

Phase E technical review result: **ready for explicit user decision**.

Approval of this document alone does not select or start a run. Any subsequent
authorization must name the exact runner label. The dependency-safe order is:

1. independently run `full10-dev42` and `heldout5-dev42`;
2. only after their selected checkpoints exist, run `composition-dev42` from
   full10 and `transfer5-dev42` from heldout5.

No command is issued by this review.
