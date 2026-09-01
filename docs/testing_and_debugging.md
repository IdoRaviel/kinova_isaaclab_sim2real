# Testing and debugging

**There is no automated test suite in this repository** (no `pytest`, no CI
test job). What exists is `test_sim/` — manual, open-loop sanity scripts you
run by hand in Isaac Sim to verify the robot/gripper physics work correctly
*before* spending time on RL training. Full detail already lives in
[`test_sim/README.md`](../test_sim/README.md); this page is a short pointer
plus the "why" for anyone deciding whether they need to run these.

## Why it exists

Debugging a broken policy is much harder when you don't know whether the
*physics* is even correct — is the gripper actually closing? Does the cube
follow the arm? Is the fingertip where you think it is? `test_sim/` isolates
those questions from RL entirely: both scripts drive the robot with raw,
scripted joint targets (no policy, no `gym.make()`, no training), so a
failure here points at the USD/physics setup, not at a reward or observation
bug.

## What's there

| Script | What it checks | Run when |
|---|---|---|
| `gripper_toggle.py` | The USD loads correctly and the arm settles at the expected hover pose (prints the EE world position after 2 s) | After changing/re-vendoring the robot USD, or before debugging any grasp-related issue, to rule out a bad asset |
| `lift.py` | The gripper actually closes and holds the cube, the cube follows the arm through a random lift (friction/contact), and env reset cleans up state | Same triggers, specifically before debugging grasp training instability |
| `test_scene_cfg.py` | Not a script to run — the shared scene definition (`ARM_HOVER_POS`, `GRIPPER_OPEN`, `GRIPPER_CLOSE`) both scripts import | Read this first if either script's output looks wrong, to see the raw joint targets being commanded |

Both scripts load the robot from
`source/gen3/gen3/assets/gen3_2f140/kinova_gen3_robotiq_2f_140.usd` — the
same canonical asset used by every training task (see
[`architecture.md`](architecture.md#robot-and-gripper-asset)).

## Prerequisites and how to run

```bash
conda activate env_isaaclab
cd test_sim
python gripper_toggle.py
# or
python lift.py
```

No pretrained checkpoints or dataset needed — these only require Isaac Sim
itself. Expected output is described per-script in
[`test_sim/README.md`](../test_sim/README.md) (e.g. `gripper_toggle.py`'s EE
height should be ~12 cm above `z=0.055`).

## If you're debugging something else

- **Debugging only the reach phase of the chain** (skip grasp/carry): set
  `reach_only = True` in `ChainCfg` — see
  [`configuration.md`](configuration.md#the-chain-scriptsrsl_rllift_and_place_cfgpy-chaincfg).
- **Debugging a single trained policy** (not the chain): `scripts/rsl_rl/play.py
  --task <id> --checkpoint <path>` — see
  [`running_and_reproduction.md`](running_and_reproduction.md), steps C/D.
- **Debugging whether a task is registered correctly**: `python
  scripts/list_envs.py`, then `python scripts/random_agent.py --task <id>` to
  confirm the env at least constructs and steps — see
  [`repository_structure.md`](repository_structure.md#scripts--everything-you-run).
