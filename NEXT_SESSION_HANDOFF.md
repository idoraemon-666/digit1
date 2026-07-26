# Next-session handoff

This is the entry point for the native-history original-protocol digit-writing
rebuild.

## Repository identity

- Working directory: `D:\digit_writing_original_protocol`.
- Working branch: `codex/original-protocol-rebuild`.
- Original-repository baseline:
  `105cd0c6b153f0e80a2593e43b7ffec7cc1f5e33`.
- Initial project commit `2c2fe1e021ab37c75db71ca8a8adce2d4b3618b0`
  is directly based on that baseline.
- Active `mRNNTorch` gitlink:
  `ac0c4f589eae37bbde63968912925de99232e306`.
- Immutable source checkout:
  `D:\digit_writing_project\reference\original_repo`.
- Historical digit-project archive:
  `D:\digit_writing_project`.

The active branch descends directly from the original paper repository. It
does not inherit the previous digit-writing implementation history.

## Reference inputs

- `PROJECT_PROTOCOL.md` is the project-local authority adapted for direct
  modification of the original repository.
- `AUTODL_SERVER_USAGE_GUIDE.md` is used only for server work.
- Local `docs/paper.pdf` is the original paper and remains ignored by Git.

Phase A does not maintain a manual whole-project checksum file. Final delivery
and actual run archives must generate hashes automatically from the files they
contain.

The user has frozen CPU as the baseline backend. Phase B and all later project
implementation, tests, audits, and eventual baseline runs are executed on the
fresh AutoDL instance with the pinned CPU environment. CUDA is not part of the
baseline result lineage unless a later protocol decision explicitly creates a
separate experiment.

## Reading order

1. `AGENTS.md`
2. `NEXT_SESSION_HANDOFF.md`
3. `PROJECT_PROTOCOL.md`
4. `AUTODL_SERVER_USAGE_GUIDE.md` only before server work

The old digit-writing brief, frozen spec, project decisions, seed-42 protocol,
implementation, tests, and reports are not part of this project.

## Active file boundary

The active tree contains:

- repository control files and the four documents listed above;
- the original training core: `config.py`, `envs.py`, `losses.py`, `model.py`,
  `train.py`, and `utils.py`;
- the pinned `mRNNTorch` submodule;
- the local original paper;
- server environment definitions and dependency inputs.

No `digit_writing/`, new-project `configurations/`, `tests/`, training runner,
or Phase A hash manifest is present yet. These paths are created incrementally
only when the corresponding implementation phase begins.

The original `analysis/`, `experiments/`, `configurations/`, and `config.txt`
remain available in Git history at commit `105cd0c...`, but are intentionally
absent from the active tree. Retrieve individual source files with `git show`
only when the protocol explicitly requires code-faithful migration. In
particular, the composition implementation must inspect
`105cd0c:experiments/exp_utils.py`, and validation/training behavior must stay
anchored to `105cd0c:train.py`, `envs.py`, `losses.py`, and `model.py`.

## Required first checks

Before editing:

```text
git status --short --branch
git rev-parse HEAD
git submodule status
git -C mRNNTorch status --short
```

Stop on unexplained changes. Never modify the immutable source checkout.

## Completed gate: Phase A rerun

The prior Phase A completion status was discarded. Phase A was rerun from the
clean commit `fa86e3ce918fc4c1ae1f6ec9304a3754b515f687` on 2026-07-26 without
running training:

1. Git ancestry and the original baseline commit were confirmed;
2. the six original training-core files remain unchanged;
3. the mRNNTorch gitlink and immutable external source boundary were confirmed;
4. full10 and heldout5 must be trained independently, with composition using
   only full10 and transfer using only heldout5;
5. no historical digit code, configuration, test, checkpoint, or result was
   imported.

The original training core and `mRNNTorch` were not modified. No checkpoint was
loaded and no training or MotorNet closed-loop run was started. Phase A added
no runtime code, configuration schema, test suite, runner, or manually
maintained checksum manifest. The CPU backend decision is an explicit user
decision recorded for subsequent server phases, not a Phase A runtime action.

## Immediate next step

After creating and verifying an auditable upload package, start Phase B of
`PROJECT_PROTOCOL.md` on the fresh AutoDL CPU server only:

1. implement the canonical primitives and digits 0-9 from sections 5 and 6;
2. implement linear arc-length timing with one join sample and C0 continuity;
3. create the single geometry/time configuration actually consumed by the
   implementation and add the required geometry and timing tests;
4. generate the prescribed-direction figures and eight direction-specific
   `2-by-5` audit figures, with each digit appearing once per figure;
5. complete the geometry, timing, and MotorNet workspace audits;
6. do not start Phase C until every Phase B gate passes.

Continue through Phases B-D only after each preceding gate is verified. Stop
at Phase E for user review before any expensive training.

## Hard stop lines

- Do not start seed-42 training, 75,000-update training, or formal seeds.
- Do not load any checkpoint or optimizer state from the historical digit
  project.
- Do not modify the immutable original-repository checkout.
- Do not add reverse stroke direction or minimum-jerk primitives.
- Do not create a preset rule-vector algebra formula.
- Do not combine the full-10 and held-out-5 base models.
- Do not change geometry, global scale, speed mapping, losses, or network size
  without an explicit protocol decision.
