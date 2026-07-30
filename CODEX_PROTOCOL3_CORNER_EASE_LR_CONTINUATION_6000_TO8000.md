# Protocol3 corner-ease six-digit LR continuation

## Scope

This is a manual, controlled continuation of the completed corner-ease v3 run at
repository HEAD `ee5a1900a33f8a8b7921fd9fc5b09aeed6600925`.

- Included digits: 0, 3, 4, 5, 6 and 7.
- Excluded digits: 1, 2 and 9 already have `STABLE_PASS`; digit 8 is reserved for
  a separate investigation.
- Each included digit resumes from its state-complete `final_checkpoint.pt` at
  update 6000.
- The source model, Adam moments, condition counts, validation history and
  Python/NumPy/Torch RNG state are restored exactly.
- The original `best_checkpoint.pt` is also supplied so the frozen three-pass
  selector can remain valid across the 6000-update boundary.

## Frozen experiment

Each digit runs two arms to exactly update 8000:

| arm | learning rate | additional updates |
|---|---:|---:|
| `lr1e3` | 0.001 | 2000 |
| `lr3e4` | 0.0003 | 2000 |

Only the Adam parameter-group learning rate may differ after optimizer-state
restoration. Batch size, gradient clipping, objective, regularization, geometry,
timing, seed, direction, delay and validation interval remain unchanged.

The twelve arms run in parallel only after the server confirms at least twelve
CPU-equivalent cores. All arms use a fixed target; there is no early stop,
automatic winner, further continuation, second seed, digit-8 run or formal
full10 run.

## Source and output isolation

The checked-in configuration freezes the SHA-256 values of the source best
checkpoint, final checkpoint and run summary for every included digit. The
server runner also verifies both source `SHA256SUMS` manifests before preparing
any continuation output.

The source v3 run is read-only. Continuation output is written under:

`runs/digit_writing_original_protocol3/scale2p50/ref100/corner_ease_v3/lr_continuation_from6000_to8000`

The selector status and best checkpoint are computed from the complete
validation history from update 0 through 8000. If the selected point remains at
or before update 6000, the copied source best checkpoint remains authoritative.

## Review boundary

The summary reports both arms without automatically selecting a winner.
Negative `3e-4 minus 1e-3` deltas favor the lower learning rate, but status,
normalized mean error, endpoint error, path-length ratio and overlay must be
reviewed together. Any engineering or safety failure invalidates the automatic
summary and stops packaging.
