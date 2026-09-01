# Extending the project

Practical guides for the changes people most often want to make. Each one
lists what's reusable, what must change, and where the current code lives —
not a promise that any of these is a one-line edit. See
[`architecture.md`](architecture.md) and [`configuration.md`](configuration.md)
for the reference detail these guides point back to.

## Quick reference: "How do I...?"

| Question | Answer is in |
|---|---|
| ...change the camera (position, resolution, lens)? | [`configuration.md`](configuration.md#camera) |
| ...use a real camera instead of the simulated one? | [`vision_pipeline.md`](vision_pipeline.md#moving-to-a-physical-camera--whats-implemented-vs-what-isnt) |
| ...change the gripper? | [`architecture.md`](architecture.md#robot-and-gripper-asset), "Swap the gripper" below |
| ...change the robot model? | [`architecture.md`](architecture.md#robot-and-gripper-asset), "Swap the robot" below |
| ...change the manipulated object (the cube)? | `self.scene.object` (a `RigidObjectCfg`) in `gen3_grasp/joint_pos_env_cfg.py` and `gen3_lift_and_place/joint_pos_env_cfg.py` — asset path, `scale`, physics props; note `collect_vision_data.py`'s `CUBE_HALF_SIZE` and the `corners_2d` keypoint projection assume this exact cube geometry, see [`configuration.md`](configuration.md#vision-dataset-collection) |
| ...add a new task? | "Add a new task" below |
| ...change the RL algorithm? | "Change the RL algorithm or framework" below, and [`configuration.md`](configuration.md#rl-hyperparameters-and-adding-another-algorithm) |
| ...change the reach→grasp handoff? | [`configuration.md`](configuration.md#the-chain-scriptsrsl_rllift_and_place_cfgpy-chaincfg), "REACH → PAUSE1 handoff" |
| ...change a pause duration? | Same page, "Pauses and lift target" |
| ...change the CARRY target range? | Same page, "CARRY workspace" |
| ...point the chain at a different checkpoint? | Same page, "Task and checkpoints" |
| ...retrain the vision model? | [`running_and_reproduction.md`](running_and_reproduction.md), steps G–K |
| ...change the vision dataset size? | [`configuration.md`](configuration.md#vision-dataset-collection) (`--num_frames`) |
| ...debug only the reach phase? | [`testing_and_debugging.md`](testing_and_debugging.md) (`ChainCfg.reach_only`) |
| ...find the final report? | [`report/kinova_project_report.pdf`](../report/kinova_project_report.pdf) |

## Add a new task

Follow the existing `gen3_*` packages as the template
(`source/gen3/gen3/tasks/manager_based/`). Concretely, for a new task
`gen3_foo`:

1. Create `gen3_foo/` with `__init__.py`, `joint_pos_env_cfg.py`, `mdp/`
   (`__init__.py` re-exporting the Isaac Lab base MDP terms you need plus any
   task-specific functions), and `agents/rsl_rl_ppo_cfg.py`.
2. In `joint_pos_env_cfg.py`, subclass the closest matching Isaac Lab
   manipulation template (`ReachEnvCfg`, `LiftEnvCfg`, or another one under
   `isaaclab_tasks/manager_based/manipulation/`) and override `scene.robot`
   (reuse `gen3.assets.KINOVA_GEN3_2F140_CFG` if the robot is unchanged),
   actions, observations, rewards, terminations, and events as needed — see
   `architecture.md`'s "inherited vs. project-added" tables for a worked
   example of exactly which fields the existing tasks override and why.
3. In `__init__.py`, call `gym.register(id="Gen3-Foo-v0", ...)` with
   `env_cfg_entry_point` and `rsl_rl_cfg_entry_point` pointing at the classes
   above (see any existing task's `__init__.py`).
4. No registry file to edit — `gen3/tasks/__init__.py`'s `import_packages`
   call auto-discovers any new `gen3_*` sub-package.
5. Verify registration with `python scripts/list_envs.py`, then sanity-check
   the env constructs and steps with `python scripts/random_agent.py --task
   Gen3-Foo-v0` *before* spending time on rewards/training.

## Change the RL algorithm or framework

Fully covered in [`configuration.md`](configuration.md#rl-hyperparameters-and-adding-another-algorithm) —
short version: algorithm selection is per-task, via named entry points
(`rsl_rl_cfg_entry_point`, `rl_games_cfg_entry_point`, ...) in that task's
`gym.register()` call, read by a framework-specific `train.py`/`play.py`. The
environment itself (observations/actions/rewards) is fully reusable across
frameworks; only the hyperparameter config and the training-loop script
differ. Remember that a checkpoint is tied to the framework that produced it.

## Swap the gripper

Covered in detail (with the exact list of what's coupled to the current
2F-140) in [`architecture.md`](architecture.md#robot-and-gripper-asset). In
short, expect to touch: the asset/USD and `KINOVA_GEN3_2F140_CFG`'s actuator
block, `FINGERTIP_OFFSET_M` and every `ee_frame` offset, every hardcoded
`finger_joint` reference (3 task configs + `test_sim/` + the chain's binary
open/close convention), and — because the action space and/or the fingertip
offset will almost certainly change — **retrain reach and grasp from
scratch**; no shipped checkpoint survives a gripper swap.

## Swap the robot

Also covered in `architecture.md`. Beyond the gripper-swap list: the arm's
joint names/count/limits (`KINOVA_GEN3_2F140_CFG.init_state`, every
`joint_names=["joint_[1-7]"]` action-term regex across three task configs),
`gen3_grasp/mdp/ik_reset.py`'s joint limits/canonical pose and its vendored
IK URDF (must be replaced to match the new arm's kinematics — the current IK
reset would silently solve for the wrong robot otherwise), and the workspace
ranges baked into every task's commands/events (tuned to the Gen3's reach
envelope; a different arm has a different one). This is the largest-blast-radius
change in the repo — treat it as a new project built on the same code
skeleton, not a config swap.

## Move toward a real camera / real hardware

Two separate gaps, both explicitly *not* implemented by this project:

- **Vision (camera calibration)** — see
  [`vision_pipeline.md`](vision_pipeline.md#moving-to-a-physical-camera--whats-implemented-vs-what-isnt)
  for the exact gap: `meta.json`'s intrinsics/extrinsics currently come only
  from the simulated `CameraCfg`; a real camera needs intrinsic + extrinsic
  calibration and, most likely, fine-tuning or retraining `CubePoseNet` on
  real images (it has only ever seen rendered frames).
- **Sim-to-real deployment (the arm itself)** — `scripts/sim2real/` exists
  but is inherited from the upstream fork this project extends, largely
  untouched by this project's own work, needs ROS2, and has at least one
  known-stale checkpoint path (see
  [`repository_structure.md`](repository_structure.md#scripts--everything-you-run)).
  The project report itself frames sim-to-real as future work. If you pick
  this up: start by fixing the stale `pretrained_models/reach` path in
  `scripts/sim2real/robots/gen3.py`, then verify from scratch whether the
  current `reach_with_orientation` policy's observation layout (it now
  includes orientation tracking, which the original reach policy this ROS2
  code was written against did not) is actually compatible with what
  `controllers/policy_controller.py` constructs and feeds to it — this has
  not been checked and may not be.
