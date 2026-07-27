# Phase C environment baseline comparison

Baseline: `105cd0c6b153f0e80a2593e43b7ffec7cc1f5e33:envs.py`.

This table records behavior preserved from the original environment and the
necessary differences introduced by `PROJECT_PROTOCOL.md`. It is audit
evidence, not a second scientific specification.

| Area | Preserved behavior | Necessary Phase C difference |
|---|---|---|
| MotorNet integration | The environment still subclasses `motornet.environment.Environment`, resets `RigidTendonArm26`, advances the effector once per `step`, and returns the original four-item step result. | Ten duplicated trajectory implementations are replaced by one shared digit environment. |
| Initial state | The original joint-space midpoint plus offsets `[0.1, 0.5]` is retained, with zero initial joint velocity. | None. |
| Trial timing | Stable and hold are 25 steps; delay is selected from 25, 50, or 75 steps; `custom_delay=150` is supported; termination remains `hold_end - 1`. | Movement length is the digit's physical arc-length interval count instead of a task-independent reach time. |
| Batch conditions | One task, speed condition, and delay are shared by the batch; each sample has its own independently selected spatial direction. | The shared task is now one digit, and directions rotate the complete canonical digit trajectory. |
| Rule input | The original ten rule columns and the original class-to-column order are retained. | Columns now mean digits 0 through 9. |
| Speed input | It is zero during stable and constant from delay through hold. | The value is determined only by the reference speed condition (`2/3`, `1/3`, or `0` in training), never by digit duration. |
| Spatial cue | It is zero during stable and present from delay through hold, with the original 0.25 m magnitude around the common anchor. | It encodes only the rigid-rotation direction and is not the digit endpoint or trajectory. |
| Go cue | It changes from zero to one at movement onset. | None. |
| Hidden target | Stable and delay hold the start; movement follows the trajectory; hold supervises the endpoint. | The movement epoch contains the arc-length interval count. Its first sample is supervised at movement onset and its final sample at hold onset, preserving the exact interval duration without dropping either endpoint. |
| Observation | Rule, speed, go cue, spatial cue, fingertip vision, and muscle proprioception remain in the original order; action history remains disabled. | The contract is explicitly checked as 28 dimensions on every observation. |
| Legacy composition | The old module name remains importable by `train.py`. | The old 20-rule composition environment is disabled because the new composition method belongs to Phase D and must not be run early. |
