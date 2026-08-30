# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluate a trained cube-pose checkpoint on a held-out dataset split.

Loads a checkpoint saved by ``train_cube_pose.py`` and reports position error (cm)
and geodesic rotation error (deg), overall and per phase. Defaults to the ``test``
split, which was never seen during training or checkpoint selection, so its numbers
are the honest final estimate of model quality.

Run from the repo root:
    python scripts/rsl_rl/vision/eval_cube_pose.py --checkpoint pretrained_models/cube_pose/best.pt
"""

import argparse

import torch
from torch.utils.data import DataLoader

from train_cube_pose import CubePoseDataset, CubePoseNet, evaluate


def main():
    parser = argparse.ArgumentParser(description="Evaluate a cube pose checkpoint.")
    parser.add_argument("--checkpoint", type=str, default="pretrained_models/cube_pose/best.pt")
    parser.add_argument("--dataset_dir", type=str, default="vision_dataset")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    train_args = ckpt["args"]
    img_size = (train_args["image_height"], train_args["image_width"])

    model = CubePoseNet(pretrained=False).to(device)
    model.load_state_dict(ckpt["model"])

    dataset = CubePoseDataset(args.dataset_dir, args.split, img_size, augment=False)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)
    print(f"[info] checkpoint {args.checkpoint} (epoch {ckpt['epoch']}), "
          f"{args.split} split: {len(dataset)} frames")

    stats = evaluate(model, loader, device)
    for phase, (pos_cm, rot_deg) in stats.items():
        print(f"  {phase:7s} {pos_cm:5.2f} cm   {rot_deg:4.1f} deg")


if __name__ == "__main__":
    main()
