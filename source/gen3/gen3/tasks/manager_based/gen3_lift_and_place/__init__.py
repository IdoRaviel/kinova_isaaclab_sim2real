"""Gymnasium environment registration for the Gen3 lift-and-place tasks.

Registers two environments:
  Gen3-LiftAndPlace-v0       — single end-to-end policy (lift + place).
  Gen3-LiftAndPlace-Chain-v0 — inference-only env that chains the frozen
                                reach_with_orientation + robust_grasp policies.
"""

import gymnasium as gym

from . import agents

gym.register(
    id="Gen3-LiftAndPlace-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:Gen3LiftAndPlaceEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Gen3LiftAndPlacePPORunnerCfg",
    },
)

gym.register(
    id="Gen3-LiftAndPlace-Chain-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:Gen3LiftAndPlaceChainEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Gen3LiftAndPlacePPORunnerCfg",
    },
)
