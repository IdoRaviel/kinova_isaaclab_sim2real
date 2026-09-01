# Configuration

Every place a numeric/behavioral value worth tuning lives, and what it
affects. Cross-referenced from [`architecture.md`](architecture.md), which
explains *why* the system behaves the way it does; this page is about
*where to change it*.

## The chain: `scripts/rsl_rl/lift_and_place_cfg.py` (`ChainCfg`)

Everything `play_lift_and_place.py` and `collect_vision_data.py` tune about
the chain lives in this one dataclass — neither script takes chain-behavior
CLI flags (only Isaac's launcher options and, for the play script, the
`--vision`/`--vision_checkpoint` switches). All fields are both training-time
*and* evaluation-time in effect — there's no separate "training config" for
the chain, it's inference-only. File for every field below:
`scripts/rsl_rl/lift_and_place_cfg.py`.

**Task and checkpoints**

| Parameter | Default | Units | Meaning | Effect | When to change |
|---|---|---|---|---|---|
| `task` | `"Gen3-LiftAndPlace-Chain-v0"` | task ID | Which registered env the chain runs | Selects the ground-truth chain env | `--vision` overrides this to the Vision variant at the CLI level; don't edit directly for that |
| `num_envs` | `1` | count | Parallel envs for the play script | More envs = more simultaneous rollouts on screen | Rarely; 1 is right for watching/recording a single rollout |
| `reach_checkpoint` / `reach_agent_cfg` | `pretrained_models/reach_with_orientation/{policy.pt,agent.yaml}` | paths | Which trained reach policy to load | Swaps the entire reach behavior | Point at a different reach checkpoint/agent-cfg pair after retraining |
| `grasp_checkpoint` / `grasp_agent_cfg` | `pretrained_models/robust_grasp/{policy.pt,agent.yaml}` | paths | Which trained grasp policy to load | Swaps the entire grasp behavior | Point at `pretrained_models/grasp/...` to use the old single-canonical-pose grasp instead of the IK-randomized retrain, if that checkpoint exists on disk (not shipped by default) |

**REACH → PAUSE1 handoff** (see `architecture.md`'s state-machine table for the hybrid logic itself)

| Parameter | Default | Units | Meaning | Effect | When to change |
|---|---|---|---|---|---|
| `reach_target_z` | `0.115` | m, base frame | Fingertip target height above the table (cube rests at `z=0.055`, so ~6 cm above it) | Where REACH aims before grasp takes over | If the cube geometry or grasp approach height changes |
| `handoff_dist` | `0.05` | m | "Close enough" — condition (a) | Smaller = REACH must get nearer before handing off | Tightening/loosening the position criterion for a clean handoff |
| `handoff_speed` | `0.03` | m/s | "Settled" — condition (a) | Smaller = fingertip must be nearly stationary | Same, for the velocity criterion |
| `handoff_accept_dist` | `0.07` | m | Only accept a plateaued handoff (b) if at least this close | Prevents handing off from far away just because progress stalled | Tune together with `handoff_plateau_steps` |
| `handoff_plateau_steps` | `30` | control steps (~1 s at this sim rate) | Steps with no distance improvement before calling it "converged" | Lower = accepts a plateau sooner | If REACH is stalling/handing off too early or too late |
| `handoff_plateau_eps` | `0.003` | m | Minimum distance improvement per step that still counts as "progress" | Threshold for what resets the plateau counter | Rarely; sensitivity tuning |
| `reach_timeout` | `120` | control steps (~4 s) | Hard fallback — end REACH regardless, condition (c) | Guarantees the episode always advances | If REACH is timing out too often (increase) or hanging too long on failure (decrease) |

**Pauses and lift target**

| Parameter | Default | Units | Meaning | Effect | When to change |
|---|---|---|---|---|---|
| `pause_after_reach` | `0.3` | **seconds** | `PAUSE1` duration | Converted to control steps at runtime via `step_dt` (`round(seconds / step_dt)`) | Shorten for a snappier demo; lengthen if the handoff needs to visibly settle |
| `pause_after_grasp` | `0.1` | **seconds** | `PAUSE2` duration | Same conversion | Same trade-off, for the post-grasp hold |
| `grasp_lift_z` | `0.155` | m, cube-center height | `GRASP`→`PAUSE2` target (cube rests at `z=0.055`, so ~10 cm above) | How high GRASP lifts before CARRY | If the cube size or desired lift clearance changes |
| `grasp_lift_tol` | `0.02` | m | `|cube_z − grasp_lift_z| ≤` this counts as "at the target" | No timeout on this phase by design — a grasp that never lifts just keeps retrying until the episode ends | Tighten/loosen how precisely the lift height must be hit |

**CARRY workspace** (kept inside the *reach* policy's trained orientation range — widening `air_pitch` past `π` or changing `air_roll` off `0` asks reach to hit poses it was never trained on)

| Parameter | Default | Units | Meaning |
|---|---|---|---|
| `air_x`, `air_y`, `air_z` | `(0.40,0.60)`, `(-0.25,0.25)`, `(0.25,0.40)` | m, base frame | Uniform sampling box for the CARRY target position |
| `air_roll` | `(0.0, 0.0)` | rad | Fixed at 0 — reach was trained with roll=0 |
| `air_pitch` | `(π/2+π/6, π)` | rad | 120° .. straight-down; never exceeds `π` (reach wasn't trained past that) |
| `air_yaw` | `(-0.3, 0.3)` | rad | Small left/right swing of the approach direction |

**Misc**

| Parameter | Default | Units | Meaning | When to change |
|---|---|---|---|---|
| `diagnostic_print_interval_steps` | `50` | control steps | How often the console diagnostic line prints (both the play script and the data collector read this same field) | Purely a logging cadence — increase for less console spam, decrease for finer-grained progress output |
| `real_time` | `False` | bool | Pace playback to match physics `dt` instead of running full speed | Watching a rollout at real-world speed instead of simulation speed |
| `reach_only` | `False` | bool | Debug switch: skip grasp/carry entirely, run only the reach phase | Debugging the reach phase in isolation within the chain env — see [`testing_and_debugging.md`](testing_and_debugging.md) |

## Per-task env configuration

Each task's `joint_pos_env_cfg.py` (or `vision_env_cfg.py`) is a
`@configclass` — Isaac Lab's own idiom for training-time configuration. These
values only take effect at the next training run or `gym.make()`; changing
them does **not** retroactively affect an already-trained checkpoint (a
checkpoint's behavior is frozen at the observation/action layout and reward
shape it was trained with — see "coupling" note at the end of this page).

- **`gen3_reach/joint_pos_env_cfg.py`** — `ee_pose` command ranges
  (`self.commands.ee_pose.ranges.*`, workspace + orientation cone), reward
  weights and tanh widths (`std=`) for position/orientation tracking,
  curriculum ramp window (`_start`/`_end` = 25%/60% of `max_iterations *
  num_steps_per_env`), payload randomization range (`mass_distribution_params`).
- **`gen3_grasp/joint_pos_env_cfg.py`** — cube spawn/IK-reset ranges
  (`self.events.reset_above_cube_ik.params`: `cube_x`, `cube_y`, `yaw_range`,
  `hover`, `jitter_deg`, `num_seeds`, `cube_jitter` — see `ik_reset.py`'s own
  docstring for units/meaning of each; see
  [`architecture.md`](architecture.md#the-ik-randomized-grasp-reset-reset_above_cube_ik)
  for the vendored URDF this reset depends on), the coarse/fine reward
  weights and `std`s (`ee_to_cube*`, `cube_to_target*`, `lifting_object`'s
  `minimal_height`), curriculum window (70% of training here, not 25–60%).
- **`gen3_lift_and_place/joint_pos_env_cfg.py`** — the base env's cube/target
  ranges (used by the never-shipped `Gen3-LiftAndPlace-v0`); the *chain* env
  subclass's episode length, arm-start joint offset range, cube-spawn range
  (intersection of grasp's and reach's trained workspaces — the in-file
  comment spells out the exact intersection), and the fixed `ee_pose`
  phase-1 target used only as a placeholder before the play script overwrites
  it every reset.
- **IK solver internals** (`gen3_grasp/mdp/ik_reset.py`) — `iters`, `damp`,
  `pos_tol`, `rot_tol`, `seed_noise`, `n_samples`, `chunk` are already named
  keyword arguments with inline comments; they tune the one-time table-build
  solve, not runtime behavior, and are unlikely to need changing unless you
  change the robot or the workspace.

## Camera

The full narrative — `meta.json` provenance, the sim-vs-real distinction, and
the complete data→train→eval→integration flow — lives in
[`vision_pipeline.md`](vision_pipeline.md). This is just the parameter
reference: defined once in `gen3_lift_and_place/joint_pos_env_cfg.py`
(`Gen3LiftAndPlaceEnvCfg.__post_init__`), part of every chain-family env's
scene, unused unless `--vision` is passed.

| Parameter | Default | Units | Meaning | When to change |
|---|---|---|---|---|
| `offset.pos` | `(1.319, 0.0, 1.153)` | m, robot-root frame (`convention="world"`, which equals robot-root here — see `architecture.md`) | Camera position | Repositioning the physical/simulated camera mount |
| `offset.rot` | `(0.0, -0.436, 0.0, 0.900)` | quaternion wxyz | ~50° down-tilt, looks at `(0.45, 0, 0.055)` | Same |
| `spawn.focal_length` | `24.0` | mm | Lens focal length | Matching a different simulated or real lens |
| `spawn.horizontal_aperture` | `20.955` | mm | Sensor width, sets horizontal FOV with focal length | Same |
| `spawn.clipping_range` | `(0.1, 4.0)` | m | Near/far render clip planes | Rarely; only if the scene extends beyond this |
| `width`, `height` | `640, 480` | pixels | Captured image resolution | Recollecting the dataset at a different native resolution (independent of `train_cube_pose.py`'s training-time resize — see `vision_pipeline.md`'s consistency check) |

## Robot and gripper

`source/gen3/gen3/assets/kinova_gen3_2f140.py` — `KINOVA_GEN3_2F140_CFG`, used
by all three trainable tasks. Full "what if I swap this" discussion:
[`architecture.md`](architecture.md#robot-and-gripper-asset).

| Parameter | Default | Units | Meaning | When to change |
|---|---|---|---|---|
| `spawn.usd_path` | `gen3/assets/gen3_2f140/kinova_gen3_robotiq_2f_140.usd` | path | The vendored robot+gripper asset | Swapping the robot or gripper model |
| `init_state.joint_pos` | `joint_1..7 = [0, 0.3, 0, 1.8, 0, 0.7, 0]`, `finger_joint = 0.0` | rad (arm), gripper units | Default/rest articulation pose | Also the IK reset's canonical-pose target (`ik_reset.py` derives `_CANONICAL` from this same value — single source of truth) |
| `actuators.arm.stiffness` | `60.0` (joints 1-4), `25.0` (joints 5-7) | N·m/rad | PD position-control gain | Retuning after a hardware/USD change; tuned for `dt=1/60 s`, `decimation=2` |
| `actuators.arm.damping` | `4.0` (1-4), `2.0` (5-7) | N·m·s/rad | PD velocity-control gain | Same |
| `actuators.gripper.stiffness` / `damping` | `37.52` / `0.00125` | — | Gripper joint PD gains | Same, if the gripper changes |
| `FINGERTIP_OFFSET_M` | `0.21` | m | Wrist flange → fingertip distance along local z | Single source of truth for every `ee_frame` offset and the grasp IK reset's `tool_offset` — changing it invalidates every trained checkpoint (see "Coupling" below) |
| Action scale (`arm_action.scale`) | `0.5` | — | Joint-position action delta scale, set independently per task's `joint_pos_env_cfg.py` | Changing action sensitivity — invalidates trained checkpoints (action semantics change) |
| Gripper open/close commands | `open_command_expr={"finger_joint": 0.0}`, `close_command_expr={"finger_joint": 0.7}` | rad | Binary gripper action mapping, set per task | Different gripper travel range or a non-binary gripper action type |

## Vision dataset collection

See [`scripts/rsl_rl/vision/README.md`](../scripts/rsl_rl/vision/README.md)
for the full CLI reference and [`vision_pipeline.md`](vision_pipeline.md) for
the narrative. `collect_vision_data.py` unless noted:

| Parameter | File | Default | Units | Meaning |
|---|---|---|---|---|
| `--num_frames` | `collect_vision_data.py` | `10000` | frames | Stop once this many frames are saved |
| `--num_envs` | `collect_vision_data.py` | `32` | count | Parallel rollouts collected from simultaneously |
| `PAUSE_SAVE_STRIDE` | `collect_vision_data.py` (module constant) | `5` | steps | Only 1-in-N `PAUSE1`/`PAUSE2` frames saved, so near-duplicate static frames don't dominate the dataset |
| `CUBE_HALF_SIZE` | `collect_vision_data.py` (module constant, derived) | `0.03 × 1.2 = 0.036` | m | Cube half-size, computed from the DexCube base asset size × the spawn scale used by the grasp/lift_and_place env configs; only feeds the (currently unused) `corners_2d` keypoint labels |
| `--num_val` / `--num_test` | `split_vision_dataset.py` | `1000` / `1000` | frames | Split sizes; remainder goes to train |
| `--seed` | `split_vision_dataset.py` | `42` | — | Shuffle seed before splitting, for reproducibility |
| Camera resolution | `CameraCfg` (see "Camera" above) | `640×480` | pixels | What's actually written to `images/` — no separate setting in the collector itself |

No seed argument on `collect_vision_data.py` itself — each collection run's
rollouts differ; the dataset, once collected, is what the deterministic
downstream steps (split/train/eval) operate on.

## Pose-network training

`train_cube_pose.py` (backbone: `CubePoseNet`, ResNet-18 + MLP head — see
[`vision_pipeline.md`](vision_pipeline.md#4-training-train_cube_posepy)):

| Parameter | Default | Units | Meaning |
|---|---|---|---|
| `--epochs` | `30` | epochs | Training length |
| `--batch_size` | `64` | samples | — |
| `--lr` | `3e-4` | — | AdamW learning rate, cosine-annealed over `--epochs` |
| `--weight_decay` | `1e-4` | — | AdamW weight decay |
| `--rot_loss_weight` | `0.1` | — | Weight of the rotation (Frobenius) loss vs. the position L2 loss |
| `--image_height` / `--image_width` | `240` / `320` | pixels | Training-time resize — independent of the camera's native 640×480 |
| `--seed` | `42` | — | `torch.manual_seed()`; GPU training still isn't bit-for-bit reproducible regardless |

Augmentation: `ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2,
hue=0.02)` on train only; both splits get `ImageNet` mean/std normalization.
Checkpoint selection: `best.pt` saved whenever val position error improves;
`last.pt` saved every epoch regardless.

`eval_cube_pose.py`: `--checkpoint` (default `pretrained_models/cube_pose/best.pt`),
`--dataset_dir` (default `vision_dataset`), `--split` (default `test`) —
deterministic, `@torch.no_grad()`, no augmentation.

## RL hyperparameters and adding another algorithm

Each task's PPO hyperparameters live in its own `agents/rsl_rl_ppo_cfg.py`
(network sizes, learning rate, clip range, entropy coefficient, etc. — see
`architecture.md`'s "per-task" tables for what the reward/observation side
looks like; this file is purely the algorithm side):

| Task | Network (actor/critic) | Iterations × steps/env |
|---|---|---|
| `Gen3-Reach-v0` | `[64, 64]` | 1500 × 24 |
| `Gen3-Grasp-v0` | `[128, 64, 64]` | 1300 × 30 |
| `Gen3-LiftAndPlace-v0` | `[256, 128, 64]` | 3000 × 24 |

**How algorithm selection actually works**: each task's `__init__.py`
registers one `gym.register()` call whose `kwargs` map a *framework name* to
a *config entry point*, e.g. (`gen3_reach/__init__.py`):

```python
kwargs={
    "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:Gen3ReachEnvCfg",
    "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Gen3ReachPPORunnerCfg",
    "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
}
```

`scripts/rsl_rl/train.py` reads `rsl_rl_cfg_entry_point`;
`scripts/rl_games/train.py` reads `rl_games_cfg_entry_point`. **Only
`Gen3-Reach-v0` has the rl_games entry point registered** — grasp and
lift-and-place only have `rsl_rl_cfg_entry_point`, so `scripts/rl_games/`
would raise a lookup error against those tasks today. rl_games itself is
present and technically functional (it's Isaac Lab's own unmodified template
script) but has only ever been exercised by this project against reach — not
project-tested on grasp or the chain.

**To add a genuinely different algorithm/framework** (e.g. SAC, or a
different PPO library): (1) write a `<framework>_cfg_entry_point` config for
the task (an Isaac-Lab-recognized hyperparameter config class/file for that
framework — see `rl_games_ppo_cfg.yaml` for the shape of a non-`rsl_rl`
example), (2) add the entry point to that task's `gym.register()` kwargs,
(3) write (or reuse an Isaac Lab template) `train.py`/`play.py` scripts that
know how to drive that framework's training-loop API — the environment
(`ManagerBasedRLEnv`, its observation/action/reward manager setup) is fully
reusable as-is; only the algorithm-side plumbing changes. A checkpoint
trained by one framework is not loadable by another's runner — `rsl_rl`'s
`OnPolicyRunner.load()` and `rl_games`' loader expect different checkpoint
formats.

## Coupling: what invalidates a trained checkpoint

Because policies are frozen at inference and chained by observation-group
matching (not retrained together), several kinds of config changes silently
break a shipped checkpoint even though nothing crashes at `gym.make()` time:
- **Observation layout** — adding/removing/reordering observation terms in a
  task's `policy` group. (This is exactly why `_ReachPolicyObsCfg` in the
  chain env exists as a byte-for-byte-matching duplicate of `ReachEnvCfg`'s
  base layout, rather than trying to reuse it directly.)
- **Action space** — dimension or semantics (e.g. the gripper's
  open/close command convention, `action ≥ 0 → open`).
- **`FINGERTIP_OFFSET_M`, robot/gripper geometry** — changes the physical
  meaning of every position observation/reward the policy was trained
  against, without changing any tensor shape (so it fails silently, not with
  an error).
- **Command ranges** (`ee_pose`/`object_pose` `ranges=...`) — a policy trained
  on one workspace range is out-of-distribution outside it; this is exactly
  why `air_pitch`/`air_roll` in `ChainCfg` are commented as "kept inside the
  reach policy's *trained* range."
- **The manipulated object itself** (cube size/mass/shape in
  `self.scene.object`) — grasp and lift_and_place were trained against this
  exact cube (size, mass, grasp affordance); changing it invalidates both
  checkpoints and requires retraining. It also invalidates the vision
  pipeline's cube-geometry assumption (`CUBE_HALF_SIZE`,
  the `corners_2d` keypoint projection) and any previously-collected
  `vision_dataset/`, which would need recollecting and `CubePoseNet`
  retraining too.

None of these are enforced by validation — a mismatch is a silent behavioral
regression, not an error, so treat "did I retrain after this change?" as a
standing question whenever you touch observations, actions, or ranges for a
task with a shipped checkpoint.
