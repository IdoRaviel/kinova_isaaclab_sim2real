from dataclasses import dataclass, field
from typing import List


@dataclass
class Gen3GraspSACCfg:
    # Training
    total_timesteps: int = 1_000_000  # total env steps before training ends
    seed: int = 42  # random seed for reproducibility
    log_name: str = "sac_gen3_grasp"  # subfolder name under logs/skrl/ or logs/sb3/
    checkpoint_freq: int = 50_000  # save a checkpoint every N steps

    # Wrapper / reward shaping (used by SB3 wrapper only)
    max_dist: float = (
        0.55  # normaliser for dist_goal only: reward = -(tanh(d_grip/0.1) + d_goal/max_dist); max cube-to-target ≈ 0.49m
    )
    success_threshold: float = 0.05  # cube is "at goal" if within 5cm → is_success=True

    # SAC hyperparameters
    buffer_size: int = (
        300_000  # max transitions stored; with 1M training steps, buffer cycles ~3.3x so stale early-policy transitions get evicted
    )
    batch_size: int = 1024  # transitions sampled per gradient update
    gradient_steps: int = (
        2  # gradient updates per env.step(); UTD = gradient_steps / num_envs ≈ 0.03 at 64 envs
    )
    gamma: float = (
        0.97  # discount factor — how much future rewards count (slightly short-sighted, good for dense rewards)
    )
    tau: float = (
        0.005  # soft target network update rate (standard SAC default; 0.05 caused Q-value explosion)
    )
    learning_rate: float = 3e-4  # Adam step size for actor, critic, and entropy
    learning_starts: int = (
        30_000  # steps of random exploration before first gradient update
    )
    action_noise_sigma: float = (
        0.1  # std of Gaussian noise added to actions during collection (SB3 only)
    )
    net_arch: List[int] = field(
        default_factory=lambda: [512, 512, 512]
    )  # hidden layer sizes for actor and critic
    n_critics: int = (
        2  # number of Q-networks; SAC takes the min to reduce overestimation
    )
    target_entropy: float = -8.0  # = -action_dim (SB3 default); -4 caused ent_coef runaway to ~20 at 122k steps
