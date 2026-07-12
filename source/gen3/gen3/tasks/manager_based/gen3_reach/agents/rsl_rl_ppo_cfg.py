# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""PPO runner configuration for the Gen3 reach task (RSL-RL).

Matched to the published pretrained checkpoint in pretrained_models/reach_with_orientation/.
"""

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class Gen3ReachPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """RSL-RL PPO configuration for Gen3-Reach-v0.

    Network: two-layer MLP [64, 64] for actor and critic.
    Training budget: 1500 iterations × 24 steps/env.
    """

    num_steps_per_env = 24
    max_iterations = 1500
    save_interval = 50
    experiment_name = "reach_gen3"
    run_name = ""
    resume = False
    empirical_normalization = False
    obs_groups = {
        "actor": ["policy"],
        "critic": ["policy"],
    }
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[64, 64],
        critic_hidden_dims=[64, 64],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.001,
        num_learning_epochs=8,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
