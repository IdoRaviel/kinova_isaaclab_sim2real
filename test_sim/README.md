# test_sim

Open-loop sanity scripts for the Kinova Gen3 + Robotiq 2F-140 setup in Isaac Sim.
These run **without any RL policy** — they use raw joint position targets to verify
that the physics, USD, and gripper mechanics are correct before training.

Both scripts load the robot from the canonical USD at:
```
source/gen3/gen3_2f140/kinova_gen3_robotiq_2f_140.usd
```

---

## Scripts

### `gripper_toggle.py`
Holds the arm in the canonical hover pose (`ARM_HOVER_POS` from `test_scene_cfg.py`)
and prints the end-effector world position after 2 seconds of settling.

Use this to verify:
- The USD loads correctly and the arm reaches the expected pose
- The EE z-height matches the expected offset above the table (~12 cm above z=0.055)

```bash
python gripper_toggle.py
```

### `lift.py`
Runs a scripted grasp-and-lift episode loop:
1. Arm settles at hover pose, gripper open (1 s)
2. Gripper closes around the cube (0.5 s)
3. Arm moves to a random joint configuration, holding the cube (2 s)
4. Scene resets, repeat

Use this to verify:
- The gripper actually closes and holds the cube
- The cube follows the arm during the random lift (friction / contact works)
- The reset cleans up state correctly

```bash
python lift.py
```

---

## Shared scene config: `test_scene_cfg.py`
Defines the scene used by both scripts: robot, DexCube, table, ground, and light.
Also exports the fixed joint targets (`ARM_HOVER_POS`, `GRIPPER_OPEN`, `GRIPPER_CLOSE`)
used across the scripts.

---

## Run from this directory
```bash
conda activate env_isaaclab
cd test_sim
python gripper_toggle.py
# or
python lift.py
```
