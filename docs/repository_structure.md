# Repository structure

A directory-by-directory map of this repo, for a reader who has read the root
[`README.md`](../README.md) and now needs to know where things actually live.
See also [`architecture.md`](architecture.md) (how the pieces interact) and
[`configuration.md`](configuration.md) (every tunable value and where it lives).

Legend used below:
- **project** — written/substantially modified for this project.
- **upstream** — inherited from Isaac Lab's own templates or from the repo
  this project forked from ([louislelay/kinova_isaaclab_sim2real](https://github.com/louislelay/kinova_isaaclab_sim2real)),
  left as-is.
- **generated** — output, not source; safe to delete and regenerate.

## Top level

| Path | Role | Category |
|---|---|---|
| `README.md` | Entry point: what this is, quick start, install, run/retrain commands. | project |
| `docs/` | This folder — structure, architecture, configuration, extension guides. | project |
| `source/gen3/` | The `gen3` Isaac Lab extension: robot asset + all task definitions (reach, grasp, lift-and-place, chain). The actual RL environments. | project (built on the Isaac Lab extension template) |
| `scripts/` | Everything you run: training/play/eval, the vision pipeline, the real-robot deployment stub, and a few generic Isaac Lab utilities. See the table below. | mixed |
| `test_sim/` | Manual, open-loop sanity scripts for the robot/gripper physics — **not** an automated test suite. Has its own [`test_sim/README.md`](../test_sim/README.md); read that for script-level detail. | project |
| `pretrained_models/` | Shipped checkpoints so the chain and vision pipeline run without retraining. See the table below. | generated (checkpoints), project (which ones are chosen/shipped) |
| `report/` | The academic project report. `kinova_project_report.pdf` is the canonical compiled document; `kinova_project_report.tex` is its LaTeX source. The report already contains a detailed narrative of the state machine, camera setup, vision architecture/results, PPO hyperparameters, and the reach-wavering issue — read it for the *why* and the experimental story; read `docs/` for the *current, operational* how-to. | project |
| `logs/rsl_rl/{grasp_gen3,reach_gen3}/<timestamp>/` | Training run outputs (checkpoints + metrics) from `scripts/rsl_rl/train.py`. No `lift_and_place_gen3` runs are present — consistent with the shipped chain using two *separately* trained reach + grasp checkpoints rather than end-to-end training. | generated |
| `outputs/<date>/<time>/` | Hydra-style run-config capture directories that Isaac Lab/RSL-RL writes automatically on every launch. | generated |
| `vision_dataset/` | The collected vision dataset (`images/`, `labels.jsonl`, `meta.json`, `{train,val,test}_labels.jsonl`). ~2.5 GB; gitignored — regenerate with `scripts/rsl_rl/vision/collect_vision_data.py` (see [`scripts/rsl_rl/vision/README.md`](../scripts/rsl_rl/vision/README.md)). | generated |
| `medias/kinova_video.mp4` | The demo video embedded in the README. | project |
| `.vscode/` | Isaac Lab's own VSCode project-template tooling (`tools/setup_vscode.py` wires up Pylance paths for Isaac Sim imports). Generic, not project-specific. | upstream |
| `.flake8`, `.pre-commit-config.yaml` | Isaac Lab's own lint/format config (black, flake8, isort, pyupgrade, codespell). Untouched framework boilerplate — note the pre-commit config still references `source/isaaclab_mimic/`, which doesn't exist in this repo, confirming it was copied wholesale from the upstream template. | upstream |
| `LICENSE` | MIT license. The copyright line still names the upstream repo's original author (Louis Le Lay) rather than this project's author, despite substantial original work since (reach/grasp retrain, the chain, the vision pipeline). Left as-is — an authorship/licensing decision, not a documentation bug. | upstream (flagged) |
| `.dockerignore` | Standard ignore list; no `Dockerfile` exists in this repo, so this file is currently vestigial. Note it also ignores `docs/` — irrelevant unless a Docker build is added later. | upstream |

## `scripts/` — everything you run

| Script | Purpose | Typical user | Category |
|---|---|---|---|
| `scripts/rsl_rl/train.py` | Train any registered task with PPO via `rsl_rl`. `--task <id>`. | Retraining reach/grasp | project-modified (Isaac Lab template + `cli_args.py` glue) |
| `scripts/rsl_rl/play.py` | Load a checkpoint and run one policy interactively (no chaining). | Sanity-checking a single trained policy | project-modified |
| `scripts/rsl_rl/eval/eval.py` | Headless, no-render evaluation of a checkpoint over many parallel envs; prints mean/std position & orientation error. Needs `sys.path.insert` to reach `cli_args.py` in its parent dir (nested one level deeper). | Quantitative before/after comparison of a retrain | project-modified |
| `scripts/rsl_rl/cli_args.py` | Shared RSL-RL CLI flag / config-loading helpers used by `train.py`, `play.py`, `eval/eval.py`. | n/a (imported, not run directly) | upstream (Isaac Lab boilerplate) |
| `scripts/rsl_rl/lift_and_place_cfg.py` | The `ChainCfg` dataclass — every tunable value for the reach→grasp→carry chain (checkpoint paths, handoff thresholds, pause durations, carry-target ranges). See [`configuration.md`](configuration.md). | Tuning the chain | project |
| `scripts/rsl_rl/play_lift_and_place.py` | **The headline demo.** Chains the frozen reach + grasp checkpoints into the full pick-and-place rollout via a per-env phase state machine. `--vision` drives it from the trained camera model instead of ground-truth cube pose. | Running/recording the main result | project |
| `scripts/rsl_rl/vision/` | The vision pipeline: `collect_vision_data.py` → `split_vision_dataset.py` → `train_cube_pose.py` → `eval_cube_pose.py`. Has its own [`README.md`](../scripts/rsl_rl/vision/README.md) with full detail — see that instead of duplicating here. | Rebuilding/retraining the camera-based pose estimator | project |
| `scripts/rl_games/train.py`, `play.py` | Alternative training/play scripts using the `rl_games` library instead of `rsl_rl`. Plain Isaac Lab template code — no project-specific logic. Only `Gen3-Reach-v0` actually has an `rl_games_cfg_entry_point` registered (see `gen3_reach/__init__.py`), so these only work for reach in practice, even though nothing in the scripts themselves enforces that. | Comparing PPO libraries on the reach task | upstream (unmodified template) |
| `scripts/list_envs.py` | Prints a table of every `Gen3-*` registered environment (id, entry point, config class). Generic, uses `gym.registry` — works for any task. | Confirming a task is registered / spelled correctly | upstream (unmodified template) |
| `scripts/random_agent.py`, `scripts/zero_agent.py` | Step a given `--task` env with random / all-zero actions; print obs/action space shapes. No trained policy involved. | Sanity-checking that an env constructs and steps at all | upstream (unmodified template) |
| `scripts/sim2real/` | ROS2 deployment code for the real Gen3 arm (`run_task_reach.py`, `controllers/policy_controller.py`, `robots/gen3.py`, `utils/config_loader.py`). **Inherited from the upstream fork** — every file's own docstring credits Louis Le Lay (`controllers/`, `robots/`) or Johnson Sun's UR10-Reacher sim2real work (`utils/config_loader.py`, which even keeps an NVIDIA copyright header, distinct from the rest of the repo). Added at repo inception and barely touched since (a formatting pass, an `rsl-rl` version-compat fix); needs `rclpy`/ROS2 to run at all. `robots/gen3.py` hardcodes `pretrained_models/reach` as the checkpoint directory, which **no longer exists** — the reach checkpoint now lives at `pretrained_models/reach_with_orientation/` after being retrained with an added orientation observation. Even if that path were repointed, whether the current policy's observation layout is still compatible with this old ROS2 node is unverified — this project's own report frames sim-to-real as future work, not a delivered result, and this code has not been exercised as part of this project. | Real-hardware deployment (not exercised by this project) | upstream (inherited, stale, unverified) |
| `test_sim/*.py` | See [`test_sim/README.md`](../test_sim/README.md). | Physics/gripper sanity checks before training | project |

### Project-specific scripts: inputs, outputs, commands

`scripts/rsl_rl/vision/*` already has this level of detail in its own
[`README.md`](../scripts/rsl_rl/vision/README.md) — not repeated here.

| File | Inputs/config | Output | Typical command |
|---|---|---|---|
| `scripts/rsl_rl/train.py` | `--task <id>`; hyperparameters from that task's `agents/rsl_rl_ppo_cfg.py` | `logs/rsl_rl/<experiment_name>/<timestamp>/{model_*.pt, agent.yaml, env.yaml}` | `python scripts/rsl_rl/train.py --task Gen3-Grasp-v0` |
| `scripts/rsl_rl/play.py` | `--task <id>`, `--checkpoint <path>` | Live render (or `--video` recording); no file output by default | `python scripts/rsl_rl/play.py --task Gen3-Reach-v0 --checkpoint pretrained_models/reach_with_orientation/policy.pt` |
| `scripts/rsl_rl/eval/eval.py` | `--task <id>`, `--checkpoint <path>`, `--num_envs`, `--num_steps` | Console-only: mean/std position & orientation error | `python scripts/rsl_rl/eval/eval.py --task Gen3-Reach-v0 --checkpoint pretrained_models/reach_with_orientation/policy.pt` |
| `scripts/rsl_rl/play_lift_and_place.py` | `ChainCfg` (`lift_and_place_cfg.py`); `--vision`/`--vision_checkpoint` | Live render of the chain; console diagnostics every `ChainCfg.diagnostic_print_interval_steps` | `python scripts/rsl_rl/play_lift_and_place.py` (add `--vision` for the camera-driven variant) |

Full step-by-step reproduction, including the vision scripts:
[`running_and_reproduction.md`](running_and_reproduction.md).

## `source/gen3/` — the Isaac Lab extension

```
source/gen3/
├── config/extension.toml         extension metadata (pip-installable package)
├── pyproject.toml, setup.py      standard Python packaging
└── gen3/
    ├── __init__.py                registers all gym environments (imports .tasks)
    ├── assets/
    │   ├── kinova_gen3_2f140.py   KINOVA_GEN3_2F140_CFG (ArticulationCfg) + FINGERTIP_OFFSET_M
    │   ├── gen3_2f140/            the robot+gripper USD asset
    │   └── urdf/gen3_2f85.urdf    arm-only URDF used by the grasp task's IK reset (see architecture.md)
    └── tasks/manager_based/
        ├── gen3_reach/            Gen3-Reach-v0
        ├── gen3_grasp/            Gen3-Grasp-v0
        └── gen3_lift_and_place/   Gen3-LiftAndPlace-v0, -Chain-v0, -Chain-Vision-v0
```

Each task package follows the same Isaac Lab convention (see
[`architecture.md`](architecture.md) for what's inherited vs. project-added
in each one):
- `__init__.py` — `gym.register()` calls (the task IDs).
- `joint_pos_env_cfg.py` (or `vision_env_cfg.py`) — the `@configclass` env
  config: scene, actions, observations, rewards, terminations, events.
- `mdp/` — custom observation/reward/termination/event/curriculum functions,
  re-exporting Isaac Lab's built-in MDP terms via `from ... import *` and
  adding only what the task needs on top.
- `agents/rsl_rl_ppo_cfg.py` — the PPO hyperparameters (network size, learning
  rate, clip range, etc.) for that task.

`source/gen3/gen3/assets/gen3_2f140/kinova_gen3_robotiq_2f_140.usd` is the
vendored robot asset — a Kinova Gen3 7-DoF arm with a Robotiq 2F-140 gripper,
consolidated here as the single source of truth for the robot geometry (an
earlier duplicate USD was removed; see `git log`). It lives alongside
`kinova_gen3_2f140.py` and `urdf/gen3_2f85.urdf` in `gen3/assets/`, so every
robot-asset file (the Python config, the sim USD, and the IK-only URDF) is
co-located in one place.

## `pretrained_models/`

| Directory | Task | Algorithm | Files | Loaded by |
|---|---|---|---|---|
| `reach_with_orientation/` | `Gen3-Reach-v0` | RSL-RL PPO | `policy.pt`, `agent.yaml`, `env.yaml` | `play.py --task Gen3-Reach-v0`, `play_lift_and_place.py` (reach phase), `scripts/sim2real/` (stale path, see above) |
| `robust_grasp/` | `Gen3-Grasp-v0` (IK-randomized-start retrain) | RSL-RL PPO | `policy.pt`, `agent.yaml`, `env.yaml` | `play.py --task Gen3-Grasp-v0`, `play_lift_and_place.py` (grasp phase) |
| `cube_pose/` | Vision pose estimator (not an RL policy) | Supervised regression (ResNet-18 + MLP head, plain PyTorch) | `best.pt` only — no `agent.yaml`/`env.yaml` since it isn't an `rsl_rl` checkpoint; the checkpoint dict itself carries `model`, `epoch`, `val_stats`, `args` | `play_lift_and_place.py --vision`, `eval_cube_pose.py`, `vision_obs.py` (env-side inference) |

`agent.yaml` + `env.yaml` are the RSL-RL runner config and full env config
dumped at training time, for reproducibility — `play.py`/`play_lift_and_place.py`
read `agent.yaml` (for `clip_actions` and the observation-group layout) but
reconstruct the env from the current task registration, not from the dumped
`env.yaml`.

There is no `pretrained_models/lift_and_place/` — `Gen3-LiftAndPlace-v0` (the
single end-to-end policy) has never been trained/shipped as part of this
project; see [`architecture.md`](architecture.md) for why.
