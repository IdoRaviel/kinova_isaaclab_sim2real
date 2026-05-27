# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Local reward functions for the Gen3 reach task.

The reach task uses only the base Isaac Lab reward functions (position_command_error,
position_command_error_tanh, orientation_command_error, action_rate_l2, joint_vel_l2),
configured in joint_pos_env_cfg.py via weight/body_name overrides on the inherited
RewardsCfg terms. No custom reward functions are required here.
"""
