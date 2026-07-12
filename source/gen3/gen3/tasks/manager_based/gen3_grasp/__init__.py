"""Gymnasium environment registration for the Gen3 grasp task.

Registers Gen3-Grasp-v0: the arm picks a cube from a random IK-initialized
start pose and places it at a random 3-D target, trained with PPO (RSL-RL).
"""

import gymnasium as gym

from . import agents

gym.register(
    id="Gen3-Grasp-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:Gen3GraspEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Gen3GraspPPORunnerCfg",
    },
)

