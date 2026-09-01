# Running and reproduction

Every command below has been checked against the actual `argparse` options and
file paths in the current repository — not copied from memory. Run all of
them from the **repo root**, with `conda activate env_isaaclab` active.

## A. Run the pretrained final chain (ground-truth cube pose)

```bash
python scripts/rsl_rl/play_lift_and_place.py
```

Loads `pretrained_models/reach_with_orientation/` and
`pretrained_models/robust_grasp/`, runs REACH→PAUSE1→GRASP→PAUSE2→CARRY in a
loop. No training or dataset required. See
[`architecture.md`](architecture.md#the-pick-and-place-chain-state-machine)
for what each phase does.

## B. Run the pretrained final chain with vision

```bash
python scripts/rsl_rl/play_lift_and_place.py --vision
```

Same as A, but every cube-pose-derived value comes from `CubePoseNet`'s
prediction on the workspace camera instead of ground truth (the true pose is
still read, for diagnostics only). Uses the shipped
`pretrained_models/cube_pose/best.pt` by default; override with
`--vision_checkpoint <path>` to use a different one (e.g. one you trained
yourself in step I).

## C. Run Reach alone

```bash
python scripts/rsl_rl/play.py --task Gen3-Reach-v0 \
    --checkpoint pretrained_models/reach_with_orientation/policy.pt
```

## D. Run Grasp alone

```bash
python scripts/rsl_rl/play.py --task Gen3-Grasp-v0 \
    --checkpoint pretrained_models/robust_grasp/policy.pt
```

First call builds a one-time IK start-state table (prints
`[ik-reset] building grasp start-state table...`) — this takes a little while
before the sim starts rendering.

## E. Retrain Reach

```bash
python scripts/rsl_rl/train.py --task Gen3-Reach-v0
```

Checkpoints/metrics → `logs/rsl_rl/reach_gen3/<timestamp>/`. Hyperparameters:
`gen3_reach/agents/rsl_rl_ppo_cfg.py`. Useful overrides: `--num_envs`,
`--max_iterations`, `--seed`, `--video` (see
[`configuration.md`](configuration.md#rl-hyperparameters-and-adding-another-algorithm)).

## F. Retrain Grasp

```bash
python scripts/rsl_rl/train.py --task Gen3-Grasp-v0
```

Checkpoints/metrics → `logs/rsl_rl/grasp_gen3/<timestamp>/`. Hyperparameters:
`gen3_grasp/agents/rsl_rl_ppo_cfg.py`.

## G. Collect a vision dataset

```bash
python scripts/rsl_rl/vision/collect_vision_data.py --output_dir vision_dataset --num_frames 10000
```

Runs the frozen reach+grasp chain (needs `pretrained_models/reach_with_orientation/`
and `pretrained_models/robust_grasp/`, same as step A) across `--num_envs`
(default 32) parallel envs and saves RGB frames + labels. Output:
`vision_dataset/{images/, labels.jsonl, meta.json}`. Full field-by-field
detail: [`vision_pipeline.md`](vision_pipeline.md).

## H. Split the dataset

```bash
python scripts/rsl_rl/vision/split_vision_dataset.py --dataset_dir vision_dataset
```

Defaults: 1000 val + 1000 test frames (`--num_val`/`--num_test`), remainder
train, shuffled with `--seed 42`. Output:
`vision_dataset/{train,val,test}_labels.jsonl`.

## I. Train the pose estimator

```bash
python scripts/rsl_rl/vision/train_cube_pose.py --dataset_dir vision_dataset
```

No Isaac Sim needed — pure PyTorch. 30 epochs by default (`--epochs`).
Output: `vision_runs/cube_pose/{last.pt, best.pt}` (`--output_dir` to change).

## J. Evaluate the pose estimator

```bash
python scripts/rsl_rl/vision/eval_cube_pose.py --checkpoint vision_runs/cube_pose/best.pt
```

Reports position/rotation error on the `test` split by default (`--split` to
change). Omitting `--checkpoint` falls back to the shipped
`pretrained_models/cube_pose/best.pt`.

## K. Run the chain with your own newly-trained vision model

```bash
python scripts/rsl_rl/play_lift_and_place.py --vision \
    --vision_checkpoint vision_runs/cube_pose/best.pt
```

Same as step B, but pointed at the checkpoint you just trained in step I
instead of the shipped one — the natural end-to-end conclusion of G→H→I→J.

---

Prerequisite checkpoints for each step, at a glance:

| Step | Needs |
|---|---|
| A, G | `pretrained_models/reach_with_orientation/`, `pretrained_models/robust_grasp/` |
| B | Above, plus `pretrained_models/cube_pose/best.pt` |
| C | `pretrained_models/reach_with_orientation/` |
| D | `pretrained_models/robust_grasp/` |
| E, F | None (trains from scratch) |
| H | Output of G |
| I | Output of H |
| J, K | Output of I (or the shipped vision checkpoint) |

See [`repository_structure.md`](repository_structure.md#pretrained_models)
for what's inside each `pretrained_models/` directory.
