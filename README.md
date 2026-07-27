# Digit-writing rebuild from the original protocol

This repository is the isolated working project for replacing the original
paper repository's ten movement tasks with parameterized digit trajectories.
Its Git ancestry starts directly from the original repository commit
`105cd0c6b153f0e80a2593e43b7ffec7cc1f5e33`.

## Current status

Phase A was rerun from clean commit `fa86e3c` on 2026-07-26. Phase B is now
complete on the AutoDL CPU baseline. It implements the protocol-defined digit
geometry, linear arc-length timing, trajectory figures, and geometry/time/
workspace audits. The accepted server head is the digit-6 correction commit
`e948eaa2e1d41f1a3258dd0ee5c6e7371b4be2bf`; all 28 server tests and the
official audit runner passed after the correction.

The earlier Phase B implementation `fd88f9b` and its run1 archive are retained
as superseded audit evidence. They were replaced because digit 6 used the
wrong traversal placement. The accepted definition starts with `D_down` to
the ellipse's upper-left tangent, then traverses the vertical-major-axis
ellipse once counterclockwise back to that tangent.

Phase C is complete at server commit
`198a850bb79607c8e5f8700bfc0215248030eb26`. The digit environment preserves
the 28-dimensional input contract and the original trial, direction, speed,
delay, and closed-loop MotorNet semantics while supporting variable-length
digit trajectories. All 33 Phase C tests passed with zero skipped, and the CPU
closed-loop smoke test passed.

The subsequent no-training protocol audit is complete at server commit
`e116b08acf452806569a59f6e9063465f1128dd3`. All 38 tests passed
with zero skipped, the complete geometry/time/workspace and rollout condition
sets were generated, and no blocking issue was found. The main non-blocking
risk is the larger spread in digit movement durations: the longest-to-shortest
movement-step ratio is 6.318, versus 2.0 for the original tasks. The audit does
not support changing the frozen geometry, speed, or timing protocol.

Phase D v2 is complete on the AutoDL CPU baseline. The main implementation
commit is `63b7bcbf913816a44c027a6e14a0c4d6f92363bd`; the runner-cleanliness
follow-up is the current server head
`b7b9b28189e33010a1ee5da28a0a27b0bd101a59`. All 51 tests passed with zero
skipped after both commits. Unlike the discarded v1 candidate, both full10
and heldout5 delegate the original
`train.py::train_subsets_base_model` rollout, five-term loss, update, and
`do_eval` logic. The two base paths still create independent models and Adam
optimizers and use separate configurations, output directories, checkpoints,
and logs.

The only RNN compatibility patch changes the call into pinned
`mRNNTorch@ac0c4f5` to its actual `(x0, input, h0=...)` signature. The original
argument order fails on the first forward pass with a deterministic dimension
error. The patch does not change the network layout, state-dict keys, weights,
or scientific protocol. Composition freezes the full10 network and optimizes
only external coefficients. Digit-5 transfer starts only from heldout5,
reinitializes input column 5, uses a fresh optimizer and strict gradient mask,
and audits every other parameter and rule column bitwise unchanged.

The first Phase D result is archived locally under
`artifacts/server-results/digit-writing-phaseD-v2-implementation-63b7bcb-run1`
with SHA-256
`61b2f9f0f8ac81465954e76e54684c2f404178cfa94bae50f0efd17b25cfa0bb`.
Its validation script stopped only after the successful commit because Python
bytecode generated inside the submodule triggered the post-commit cleanliness
gate; the retained test evidence was verified and the archive step was resumed
without rerunning tests. The follow-up prevents bytecode writes during formal
experiments and rechecks both repositories after a run. Its independently
verified result is under
`artifacts/server-results/digit-writing-phaseD-v2.1-runner-cleanliness-b7b9b28-run1`
with SHA-256
`449810f67e309826c0e4fba15c69ec76bea85f31fe5ac42f8cc2b4d0f1c1cded`.
No development seed, base training, composition, or transfer was started.

The Phase D v1 upload package remains obsolete and must not be uploaded or
run. It was never applied to the server and no training was started.

CPU is the user-selected baseline backend. Phase B and all later project work
are executed on the AutoDL instance using the pinned CPU environment;
CUDA is not part of the baseline result lineage.

Do not run training from this checkout until the implementation and gates in
`PROJECT_PROTOCOL.md` have been completed and reviewed.

## Authority and source boundaries

- `PROJECT_PROTOCOL.md` is the sole scientific and implementation authority.
- The immutable source checkout remains at
  `D:\digit_writing_project\reference\original_repo`.
- `mRNNTorch` is pinned to the exact gitlink used by the source commit.
- `docs/paper.pdf` is a local, Git-ignored copy of the original paper.
- The previous digit-writing project at `D:\digit_writing_project` remains the
  historical development and audit archive; none of its digit implementation,
  checkpoints, tests, reports, or configurations are imported here.

## Active layout

- `envs.py`: accepted Phase C digit environment built as a minimal adaptation
  of the original environment and 28-dimensional input contract.
- `train.py`, `model.py`, `config.py`, `losses.py`, `utils.py`: inherited
  training core plus the minimal Phase D v2 adapters, checkpoint metadata,
  compatibility fix, and transfer-column isolation.
- `mRNNTorch/`: pinned RNN dependency.
- `digit_writing/geometry.py`: canonical primitives, digits, rotations, and
  linear arc-length timing.
- `digit_writing/geometry_audit.py`: figure generation and workspace audits.
- `digit_writing/protocol_audit.py`: no-training geometry, timing, workspace,
  rollout, and original-task comparison audit.
- `digit_writing/experiments.py`: frozen-network composition adapter and the
  thin digit-5 transfer entry.
- `configurations/digit_original_protocol_geometry.json`: the Phase B-only
  geometry/time configuration.
- `configurations/digit_original_protocol_*.json`: four Phase D v2
  configurations in addition to the accepted geometry configuration.
- `tests/`: accepted Phase B, Phase C, and protocol-audit tests, plus Phase D
  original-core reuse, compatibility, freeze, gradient, and runner tests.
- `server/`: pinned CPU environment inputs, accepted audit runners, and scoped
  Phase D runners that require the exact experiment label as Phase E
  authorization.
- `NEXT_SESSION_HANDOFF.md`: exact continuation point and hard stop lines.
- `PHASE_E_DELIVERY_REVIEW.md`: protocol-mandated delivery checklist,
  evidence index, risk register, and explicit authorization boundary.

Original analysis and experiment scripts are absent from the active tree to
keep the project focused. They remain byte-exact in Git history at the source
commit and can be inspected with `git show` when a protocol step explicitly
requires them.

Phase E delivery review is now prepared in `PHASE_E_DELIVERY_REVIEW.md` against
server head `b7b9b28...` and the two independently verified Phase D result
archives. Its technical finding is ready for explicit user decision with no
implementation blocker found. This is not a run authorization: a development
seed, both 75,000-update base runs, composition optimization, and digit-5
transfer remain prohibited until the user explicitly authorizes the exact run
label.
