# Vision-based cube pose estimation

Scripts for replacing the chain's ground-truth cube pose with an estimate from a
fixed workspace RGB camera, so `play_lift_and_place.py` doesn't depend on privileged
simulator state. Background/motivation: see the "Vision-based cube pose estimation"
section of the top-level `README.md`.

## Pipeline

```
collect_vision_data.py  -->  split_vision_dataset.py  -->  train_cube_pose.py  -->  eval_cube_pose.py
   (needs Isaac Sim)             (pure Python)             (pure PyTorch)          (pure PyTorch)
        |                                                        |
        v                                                        v
  vision_dataset/                                    pretrained_models/cube_pose/best.pt
  (images/ + labels.jsonl)                                       |
        |                                                        v
        +----------------------------------->  play_lift_and_place.py --vision
                                                (scripts/rsl_rl/, consumes the checkpoint)
```

Run in order, from the repo root:

```bash
python scripts/rsl_rl/vision/collect_vision_data.py --output_dir vision_dataset --num_frames 10000
python scripts/rsl_rl/vision/split_vision_dataset.py --dataset_dir vision_dataset
python scripts/rsl_rl/vision/train_cube_pose.py --dataset_dir vision_dataset
python scripts/rsl_rl/vision/eval_cube_pose.py --checkpoint vision_runs/cube_pose/best.pt
```

Only the first step needs Isaac Sim; the rest are plain PyTorch/stdlib and can run
anywhere with the dataset on disk (e.g. a machine without Isaac Lab installed).

## Files

### `collect_vision_data.py`
Runs the same frozen reach → grasp → carry chain as `../play_lift_and_place.py`
(shares its config, `../lift_and_place_cfg.py::CFG`) across parallel envs, headless,
and instead of rendering for viewing, saves the workspace camera's RGB frame plus
ground-truth cube-pose labels at each step — until `--num_frames` frames are saved.
Needs Isaac Sim + `rsl_rl` (loads the frozen reach/grasp checkpoints via
`OnPolicyRunner` to drive the rollout).

- Subsamples near-static PAUSE1/PAUSE2 frames (1 in `PAUSE_SAVE_STRIDE`) and stops
  saving once a CARRY episode has settled, so the dataset isn't dominated by
  near-duplicate static frames.
- Writes, under `--output_dir`:
  - `images/NNNNNN.png` — RGB frames, at the camera's native resolution.
  - `labels.jsonl` — one JSON record per saved frame: `file`, `env_id`, `step`,
    `phase` (`reach`/`grasp`/`carry`/...), `cube_pos` + `cube_quat_wxyz` (robot-root
    frame — the labels `train_cube_pose.py` regresses against), and `corners_2d`
    (the cube's 8 corners projected to pixel coordinates — saved for a possible
    future keypoint+PnP model, unused by the current direct-regression model).
  - `meta.json` — camera intrinsics/extrinsics, written once (constant across frames).
- Key args: `--output_dir`, `--num_frames`, `--num_envs`.

### `split_vision_dataset.py`
Shuffles `labels.jsonl` (fixed seed, for reproducibility) and splits it into
`train_labels.jsonl` / `val_labels.jsonl` / `test_labels.jsonl` inside the same
`--dataset_dir`. Frames are collected in rollout order (every env starts each
episode in REACH at once), so a sequential split would badly skew each split's
phase mix — shuffling first ensures train/val/test all get a representative sample
of every phase. Images in `images/` are untouched; the split files just reference
the same `file` paths as `labels.jsonl`. Prints each split's size and phase
distribution. Key args: `--dataset_dir`, `--num_val`, `--num_test`, `--seed`.

### `train_cube_pose.py`
Trains `CubePoseNet` — an ImageNet-pretrained ResNet-18 backbone with a small MLP
head — to regress the cube's 3-D position (robot-root frame) and full 3-DoF
orientation from a single RGB frame. Orientation is output as a continuous 6-D
rotation representation (Zhou et al.) rather than a quaternion, to avoid the
antipodal sign ambiguity that makes quaternions harder to regress; it's decoded to
a rotation matrix via Gram-Schmidt. Reads `{train,val}_labels.jsonl` +
`images/` written by the two scripts above.

- Each epoch: trains one pass, then evaluates on `val`, reporting position error
  (cm) and geodesic rotation error (deg) overall and per phase.
- Saves `last.pt` every epoch and `best.pt` (by val position error) under
  `--output_dir` (default `vision_runs/cube_pose`); the checkpoint used by the
  chain, `pretrained_models/cube_pose/best.pt`, is a copy of a `best.pt` from a
  training run.
- Also defines `CubePoseDataset`, `CubePoseNet`, and `evaluate()`, imported directly
  by `eval_cube_pose.py` (they're not duplicated there).
- Key args: `--dataset_dir`, `--output_dir`, `--epochs`, `--batch_size`, `--lr`,
  `--rot_loss_weight`, `--image_height`/`--image_width` (frames are resized to this
  for training — independent of the raw resolution saved by the collector).
- No Isaac Sim needed — pure PyTorch/torchvision.

### `eval_cube_pose.py`
Loads a checkpoint saved by `train_cube_pose.py` and reports position/rotation error
on one dataset split — defaults to `test`, which is never used for training or
checkpoint selection, so it's the honest final accuracy number (this is where the
0.50 cm / 1.2° figures in the top-level README come from). Key args: `--checkpoint`,
`--dataset_dir`, `--split` (`train`/`val`/`test`). No Isaac Sim needed.

## Where this plugs back in

`../play_lift_and_place.py --vision` loads `pretrained_models/cube_pose/best.pt` and
uses `CubePoseNet`'s live prediction on the camera image everywhere the chain would
otherwise use ground-truth cube pose (grasp policy's object observations, reach
target, lift target, the GRASP→PAUSE2 transition check). The true pose is then only
read for diagnostics (a periodic prediction-error print and a per-episode success
count) — see that script's module docstring for the full detail.
