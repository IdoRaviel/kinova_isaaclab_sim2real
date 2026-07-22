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

<video src="medias/kinova_video.mp4" controls width="600"></video>

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

Currently, the cube's location used to set the reach/grasp targets is computed
via IK from ground-truth simulator state; vision-based pose estimation from a
workspace camera is in progress on a separate branch.

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
`scripts/rsl_rl/lift_and_place_cfg.py`.

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
