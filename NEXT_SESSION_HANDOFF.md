# Next-session handoff

This is the entry point for the native-history original-protocol digit-writing
rebuild.

## Repository identity

- Working directory: `D:\digit_writing_original_protocol`.
- Working branch: `codex/original-protocol-rebuild`.
- Direct original-repository parent:
  `105cd0c6b153f0e80a2593e43b7ffec7cc1f5e33`.
- Active `mRNNTorch` gitlink:
  `ac0c4f589eae37bbde63968912925de99232e306`.
- Immutable source checkout:
  `D:\digit_writing_project\reference\original_repo`.
- Historical digit-project archive:
  `D:\digit_writing_project`.

The active branch descends directly from the original paper repository. It
does not inherit the previous digit-writing implementation history.

## Imported local inputs

- `PROJECT_PROTOCOL.md` is an exact copy of
  `codex_digit_writing_original_protocol_rebuild.md` with SHA-256
  `5dba67ea66ae4d48cc2c60bda65ba6d783e546749a8596f4550c143eb6876290`.
- `AUTODL_SERVER_USAGE_GUIDE.md` preserves the source working-tree version with
  SHA-256
  `40d907887f553a6b85959d32d9f72644a74bec6059b732a63851ff87ac5c5654`.
- Local `docs/paper.pdf` has SHA-256
  `c8f69932abc8f4ed9a40af686f2041a427c2fe958181f5177eea24c2f17d40ad`
  and remains ignored by Git.

## Reading order

1. `AGENTS.md`
2. `NEXT_SESSION_HANDOFF.md`
3. `PROJECT_PROTOCOL.md`
4. `AUTODL_SERVER_USAGE_GUIDE.md` only before server work

The old digit-writing brief, frozen spec, project decisions, seed-42 protocol,
implementation, tests, and reports are not part of this project.

## Active file boundary

The active tree intentionally contains only:

- repository control files and the four documents listed above;
- the original training core: `config.py`, `envs.py`, `losses.py`, `model.py`,
  `train.py`, and `utils.py`;
- the pinned `mRNNTorch` submodule;
- the local original paper;
- server environment definitions and dependency inputs.

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

## Immediate next step

Start Phase A of `PROJECT_PROTOCOL.md` only:

1. map each retained original behavior to the new configuration schema;
2. create the new digit project modules and tests without editing unrelated
   original logic;
3. remove reverse-stroke, minimum-jerk, velocity-loss, custom behavior-score,
   and preset-algebra assumptions by not importing them from the historical
   digit project;
4. do not run any training.

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

