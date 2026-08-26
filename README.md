# Kinova Gen3 RL Pick-and-Place

> Final project for Bar-Ilan University's AI-Excellence Program, developed in
> collaboration with a Bar-Ilan University research lab.

Reinforcement learning pipeline for the **Kinova Gen3** robot arm, trained in
[Isaac Lab](https://github.com/isaac-sim/IsaacLab).

The headline result is a **pick-and-place chain**: two policies I trained
separately — a *reach* policy and a *grasp* policy — are frozen and chained
together at inference through **three phases**: reach first drives the
gripper top-down above the cube, then grasp picks it up and lifts it, then
reach takes back over to carry the cube to a target pose in the air.

> Pre-trained models are included — you can run the full chain immediately
> without training.

<video src="https://github.com/user-attachments/assets/b1c831af-7756-4d87-803d-62f1a6ef21f7" controls width="600"></video>

*Target markers: **blue** = target pose for the first reach (above the cube),
**green** = target pose for the second reach (carry, in the air).*

---

## Background

- **Deep RL**: both policies are trained with continuous-control PPO,
  observing joint state + pose commands and outputting joint position
  targets (+ gripper command).
- **Algorithm**: PPO via `rsl_rl`, trained per-task in Isaac Lab's
  GPU-parallel simulation, then frozen and composed at play-time.
- **Reward shaping**: reward terms were iteratively designed and tuned per
  task (e.g. distance-to-target terms, a lifting-the-object bonus for grasp,
  action-rate/velocity penalties) to shape behavior and break local optima,
  such as hovering near the cube without ever grasping it.
- **Training monitoring**: progress was tracked live from the training
  script's console output, printed each iteration — mean total reward, a
  per-term reward breakdown, PPO losses, entropy, and other diagnostics —
  used to judge convergence and compare runs.

By default the cube's location used to set the reach/grasp targets is read
from ground-truth simulator state; this branch also adds a **vision-based
alternative** — see the next section.

---

## Vision-based cube pose estimation

**Motivation.** In simulation the cube's pose comes for free: the chain reads it
directly from privileged simulator state to compute the reach and grasp targets.
A real robot has no such oracle — the cube's pose must be *perceived*. This
branch replaces the ground-truth cube pose with an estimate from a fixed
workspace RGB camera, removing the chain's dependence on privileged state and
taking a necessary step toward running it on real hardware.

**Setup.** A fixed, elevated, oblique-view camera is added to the chain
environment, placed beyond the cube's reachable range and centered laterally so
the descending gripper never fully occludes the cube.

### 1. Collect a dataset from chain rollouts

```bash
python scripts/rsl_rl/vision/collect_vision_data.py --output_dir vision_dataset --num_frames 10000
python scripts/rsl_rl/vision/split_vision_dataset.py --dataset_dir vision_dataset
```

The collector runs the same frozen reach → grasp → carry chain as the play
script across parallel envs and saves, at each step, the camera's RGB frame
together with ground-truth labels. Near-static pause frames are subsampled and
settled carry frames skipped, so the dataset isn't dominated by duplicates. The
splitter then shuffles (fixed seed) and splits into train/val/test (1000 val +
1000 test frames, remainder train), so each split gets a representative mix of
phases.

Dataset layout (`vision_dataset/`):

```
images/NNNNNN.png                RGB frames (320x240)
labels.jsonl                     one JSON record per frame:
                                   file, env_id, step, phase,
                                   cube_pos (robot-root frame, m),
                                   cube_quat_wxyz (robot-root frame),
                                   corners_2d (8 cube corners, pixels)
meta.json                        camera intrinsics + fixed extrinsics
{train,val,test}_labels.jsonl    split label files
```

The 2-D corner keypoints and camera geometry are saved to keep a keypoint+PnP
variant possible without re-collecting; the current model doesn't use them.

### 2. Train and evaluate the pose model

```bash
python scripts/rsl_rl/vision/train_cube_pose.py --dataset_dir vision_dataset
python scripts/rsl_rl/vision/eval_cube_pose.py    # test split by default
```

`CubePoseNet` (a ResNet-18 backbone with an MLP head) regresses the cube's
3-D position and orientation (6-D rotation representation) directly from a
single RGB frame. Trained for 30 epochs; the best checkpoint by validation
position error is included at `pretrained_models/cube_pose/best.pt`.

**Results on the held-out test split (1000 frames): 0.50 cm position error,
1.2° rotation error.** Accuracy is best in the phases that matter most — while
the cube sits unoccluded on the table, before grasping (~0.3–0.4 cm) — and
worst right after the lift, when the closed gripper partially hides the cube
(~1 cm), still small relative to the 7.2 cm cube.

### 3. Run the chain on vision

```bash
python scripts/rsl_rl/play_lift_and_place.py --vision
```

Same chain as the default play script, but the grasp policy's object
observations and every cube-pose-derived target/transition are computed from
`CubePoseNet`'s prediction on the live camera image instead of the simulator's
ground-truth cube pose (which is then used only for diagnostics: a periodic
prediction-error print and a per-episode success count).

---

## Installation

1. Install Isaac Lab following the official
   [installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html)
   (conda recommended).
2. Clone this repo **outside** the `IsaacLab` directory.
3. Install the package:

```bash
conda activate env_isaaclab
python -m pip install -e source/gen3
```

All scripts below must be run from the **repo root**.

---

## Run the full pick-and-place chain

```bash
python scripts/rsl_rl/play_lift_and_place.py
```

Loads the pre-trained reach and grasp policies from `pretrained_models/` and
runs the reach → pause → grasp → carry sequence. Tunable settings (handoff
thresholds, phase timeouts, carry target range) live in
`scripts/rsl_rl/lift_and_place_cfg.py`. Add `--vision` to drive the chain from
the camera instead of ground-truth cube pose (see the vision section above).

---

## Run each policy separately

**Reach** (fingertip to a random top-down goal pose):

```bash
python scripts/rsl_rl/play.py --task Gen3-Reach-v0 \
    --checkpoint pretrained_models/reach_with_orientation/policy.pt
```

**Grasp** (pick cube from a random IK-initialized start, place at a 3D target):

```bash
python scripts/rsl_rl/play.py --task Gen3-Grasp-v0 \
    --checkpoint pretrained_models/robust_grasp/policy.pt
```

---

## Retrain a policy

**Reach:**

```bash
python scripts/rsl_rl/train.py --task Gen3-Reach-v0
```

**Grasp:**

```bash
python scripts/rsl_rl/train.py --task Gen3-Grasp-v0
```

Checkpoints and metrics are saved to `logs/rsl_rl/<experiment_name>/`.
The reach task also supports training with rl_games:

```bash
python scripts/rl_games/train.py --task Gen3-Reach-v0
```

---

## Acknowledgements

This repository extends
[louislelay/kinova_isaaclab_sim2real](https://github.com/louislelay/kinova_isaaclab_sim2real)
with a full pick-and-place chain and a retrained robust grasp policy.

- [Isaac Lab](https://github.com/isaac-sim/IsaacLab) — simulation framework
- [INIT Lab](https://initrobots.ca/) — David St-Onge, Augustin Nguon
- Johnson Sun — [UR10 Reacher sim2real](https://github.com/j3soon/OmniIsaacGymEnvs-UR10Reacher)
