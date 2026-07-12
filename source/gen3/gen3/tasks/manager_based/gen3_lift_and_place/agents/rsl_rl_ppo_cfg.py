"""PPO runner configuration for the Gen3 lift-and-place task (RSL-RL).

Used for training Gen3-LiftAndPlace-v0 (single end-to-end policy).
Not used during chain inference — the chain loads reach and grasp
checkpoints separately via lift_and_place_cfg.py.
"""

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class Gen3LiftAndPlacePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """RSL-RL PPO configuration for Gen3-LiftAndPlace-v0.

    Network: three-layer MLP [256, 128, 64] for actor and critic.
    Training budget: 3000 iterations × 24 steps/env.
    """

    num_steps_per_env = 24
    max_iterations = 3000
    save_interval = 50
    experiment_name = "lift_and_place_gen3"
    run_name = ""
    resume = False
    clip_actions = 1.0
    empirical_normalization = True
    obs_groups = {
        "actor": ["policy"],
        "critic": ["policy"],
    }
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.006,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=5.0e-5,
        schedule="adaptive",
        gamma=0.98,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
