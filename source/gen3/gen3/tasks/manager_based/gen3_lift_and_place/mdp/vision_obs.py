"""Vision-based cube-pose observation terms for the Gen3 lift-and-place chain.

Drop-in replacements for the privileged ``object_position_in_robot_root_frame`` /
``object_orientation_in_robot_root_frame`` observation terms: instead of reading the
simulator's ground-truth object state, they run the trained ``CubePoseNet`` regressor
(see ``scripts/rsl_rl/vision/train_cube_pose.py``) on the workspace camera's RGB frame
and return the *predicted* cube pose in the robot-root frame.

Both terms share one network forward per env step: the first term called runs
inference and caches the result on the env (keyed by ``common_step_counter``); the
second reuses it. The cached prediction is also left on the env as
``env._vision_pose_cache`` so the play script's state machine (target setting, lift
detection) can consume the same prediction instead of privileged state.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18

from isaaclab.utils.math import quat_from_matrix

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def _rot6d_to_matrix(x: torch.Tensor) -> torch.Tensor:
    """Decode a (N, 6) 6-D rotation into (N, 3, 3) via Gram-Schmidt."""
    a1, a2 = x[..., :3], x[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = F.normalize(a2 - (b1 * a2).sum(-1, keepdim=True) * b1, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-1)


class _CubePoseNet(nn.Module):
    """ResNet-18 + MLP head -> 3 position + 6-D rotation.

    Architecture must match ``CubePoseNet`` in scripts/rsl_rl/vision/train_cube_pose.py
    exactly (the checkpoint's state_dict is loaded into this class).
    """

    def __init__(self):
        super().__init__()
        backbone = resnet18(weights=None)
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.head = nn.Sequential(nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, 9))

    def forward(self, img: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.head(self.backbone(img))
        return out[..., :3], _rot6d_to_matrix(out[..., 3:])


def _predict(env: ManagerBasedRLEnv, checkpoint: str) -> dict:
    """Run (or reuse this step's) cube-pose inference; returns {step, pos, quat}."""
    step = int(env.common_step_counter)
    cache = getattr(env, "_vision_pose_cache", None)
    if cache is not None and cache["step"] == step:
        return cache

    model = getattr(env, "_vision_pose_model", None)
    if model is None:
        ckpt = torch.load(checkpoint, map_location=env.device, weights_only=False)
        model = _CubePoseNet().to(env.device)
        model.load_state_dict(ckpt["model"])
        model.eval()
        env._vision_pose_model = model
        env._vision_pose_imgsize = (
            ckpt["args"]["image_height"],
            ckpt["args"]["image_width"],
        )
        mean = torch.tensor(_IMAGENET_MEAN, device=env.device).view(1, 3, 1, 1)
        std = torch.tensor(_IMAGENET_STD, device=env.device).view(1, 3, 1, 1)
        env._vision_pose_norm = (mean, std)
        print(f"[vision_obs] loaded cube-pose net from {checkpoint} (epoch {ckpt['epoch']})")

    rgb = env.scene["camera"].data.output["rgb"][..., :3]  # (N, H, W, 3) uint8
    img = rgb.permute(0, 3, 1, 2).float() / 255.0
    img = F.interpolate(
        img, size=env._vision_pose_imgsize, mode="bilinear",
        align_corners=False, antialias=True,
    )
    mean, std = env._vision_pose_norm
    img = (img - mean) / std
    with torch.no_grad():
        pos, rot = model(img)

    cache = {"step": step, "pos": pos, "quat": quat_from_matrix(rot)}
    env._vision_pose_cache = cache
    return cache


def object_position_from_vision(
    env: ManagerBasedRLEnv, checkpoint: str = "pretrained_models/cube_pose/best.pt"
) -> torch.Tensor:
    """Predicted cube position in the robot-root frame (N, 3), from the camera image."""
    return _predict(env, checkpoint)["pos"]


def object_orientation_from_vision(
    env: ManagerBasedRLEnv, checkpoint: str = "pretrained_models/cube_pose/best.pt"
) -> torch.Tensor:
    """Predicted cube orientation in the robot-root frame (N, 4 wxyz), from the camera image."""
    return _predict(env, checkpoint)["quat"]
