# Digit-writing rebuild from the original protocol

This repository is the isolated working project for replacing the original
paper repository's ten movement tasks with parameterized digit trajectories.
Its Git ancestry starts directly from the original repository commit
`105cd0c6b153f0e80a2593e43b7ffec7cc1f5e33`.

## Current status

The active Python files are the untouched original training core. They still
implement the paper's original movement tasks and are retained only as the
behavioral baseline for the rebuild. The lightweight Phase A identity check is
complete: Git ancestry, the original training core, the mRNNTorch gitlink, the
external archive boundary, and the required full10/heldout5 separation were
confirmed. Phase A intentionally added no runtime package, configuration,
tests, runner, or device restriction. Geometry generation, environments,
training code, and runners have not been implemented yet.

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

- `envs.py`, `train.py`, `losses.py`, `model.py`, `utils.py`, `config.py`:
  original paper repository training core.
- `mRNNTorch/`: pinned RNN dependency.
- `server/`: environment construction files only.
- `NEXT_SESSION_HANDOFF.md`: exact continuation point and hard stop lines.

The `digit_writing/`, `configurations/`, and `tests/` paths will be created
incrementally when their corresponding implementation phase begins. They are
not pre-populated during Phase A.

Original analysis and experiment scripts are absent from the active tree to
keep the project focused. They remain byte-exact in Git history at the source
commit and can be inspected with `git show` when a protocol step explicitly
requires them.

The next implementation step is Phase B in `PROJECT_PROTOCOL.md`: geometry,
linear arc-length timing, trajectory figures, and geometry/time/workspace
audits. No training is authorized.
