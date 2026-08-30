# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Train a CNN to regress the cube's 6-DoF pose from a single RGB frame.

Direct-regression baseline for the vision pipeline: an ImageNet-pretrained ResNet-18
backbone with a small head that outputs the cube's 3-D position (robot-root frame,
meters) plus its orientation as a 6-D rotation representation (Zhou et al., "On the
Continuity of Rotation Representations in Neural Networks") -- the first two columns
of the rotation matrix, re-orthonormalized with Gram-Schmidt at decode time. 6-D is
used instead of quaternions because it is continuous (no antipodal sign ambiguity),
which regresses better. Full 3-DoF orientation is needed: once gripped, the cube
tilts freely (84% of CARRY frames are non-yaw-only), and the distinct face colors
make full orientation observable.

Reads the dataset written by ``collect_vision_data.py`` / ``split_vision_dataset.py``
(``{train,val}_labels.jsonl`` + ``images/``). Reports position error (cm) and geodesic
rotation error (deg) on val each epoch, broken down per phase (reach/grasp/carry), and
saves the best checkpoint by val position error.

Run from the repo root (no Isaac Sim needed -- pure PyTorch):
    python scripts/rsl_rl/vision/train_cube_pose.py --dataset_dir vision_dataset
"""

import argparse
import json
import math
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import ResNet18_Weights, resnet18

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def quat_wxyz_to_matrix(q: torch.Tensor) -> torch.Tensor:
    """Convert (N, 4) wxyz quaternions to (N, 3, 3) rotation matrices."""
    w, x, y, z = q.unbind(-1)
    return torch.stack(
        [
            1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
            2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
            2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
        ],
        dim=-1,
    ).reshape(q.shape[:-1] + (3, 3))


def rot6d_to_matrix(x: torch.Tensor) -> torch.Tensor:
    """Decode a (N, 6) 6-D rotation into (N, 3, 3) via Gram-Schmidt."""
    a1, a2 = x[..., :3], x[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = F.normalize(a2 - (b1 * a2).sum(-1, keepdim=True) * b1, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-1)


def geodesic_deg(r_pred: torch.Tensor, r_gt: torch.Tensor) -> torch.Tensor:
    """Per-sample geodesic angle (degrees) between two batches of rotation matrices."""
    rel = r_pred.transpose(-1, -2) @ r_gt
    cos = (rel.diagonal(dim1=-2, dim2=-1).sum(-1) - 1.0) / 2.0
    return torch.rad2deg(torch.acos(cos.clamp(-1.0, 1.0)))


class CubePoseDataset(Dataset):
    """RGB frame -> (position, rotation matrix) pairs from a labels.jsonl split."""

    def __init__(self, dataset_dir: str, split: str, image_size: tuple[int, int], augment: bool):
        with open(os.path.join(dataset_dir, f"{split}_labels.jsonl")) as f:
            self.records = [json.loads(line) for line in f]
        self.dataset_dir = dataset_dir
        ops = [transforms.Resize(image_size)]
        if augment:
            ops.append(transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02))
        ops += [transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
        self.transform = transforms.Compose(ops)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        img = Image.open(os.path.join(self.dataset_dir, rec["file"])).convert("RGB")
        pos = torch.tensor(rec["cube_pos"], dtype=torch.float32)
        rot = quat_wxyz_to_matrix(torch.tensor(rec["cube_quat_wxyz"], dtype=torch.float32))
        return self.transform(img), pos, rot, rec["phase"]


class CubePoseNet(nn.Module):
    """ResNet-18 backbone + MLP head -> 3 position + 6-D rotation outputs."""

    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = resnet18(weights=weights)
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.head = nn.Sequential(nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, 9))

    def forward(self, img: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.head(self.backbone(img))
        return out[..., :3], rot6d_to_matrix(out[..., 3:])


@torch.no_grad()
def evaluate(model, loader, device):
    """Return overall and per-phase mean position error (cm) and rotation error (deg)."""
    model.eval()
    pos_err, rot_err, phases = [], [], []
    for img, pos, rot, phase in loader:
        img = img.to(device, non_blocking=True)
        pred_pos, pred_rot = model(img)
        pos_err.append((pred_pos.cpu() - pos).norm(dim=-1) * 100.0)
        rot_err.append(geodesic_deg(pred_rot.cpu(), rot))
        phases += list(phase)
    pos_err, rot_err = torch.cat(pos_err), torch.cat(rot_err)
    stats = {"all": (pos_err.mean().item(), rot_err.mean().item())}
    for ph in sorted(set(phases)):
        mask = torch.tensor([p == ph for p in phases])
        stats[ph] = (pos_err[mask].mean().item(), rot_err[mask].mean().item())
    return stats


def main():
    parser = argparse.ArgumentParser(description="Train the cube pose regressor.")
    parser.add_argument("--dataset_dir", type=str, default="vision_dataset")
    parser.add_argument("--output_dir", type=str, default="vision_runs/cube_pose")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--rot_loss_weight", type=float, default=0.1,
                        help="Weight of the rotation (Frobenius) loss vs the position L2 loss.")
    parser.add_argument("--image_height", type=int, default=240)
    parser.add_argument("--image_width", type=int, default=320)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)
    img_size = (args.image_height, args.image_width)

    train_ds = CubePoseDataset(args.dataset_dir, "train", img_size, augment=True)
    val_ds = CubePoseDataset(args.dataset_dir, "val", img_size, augment=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)
    print(f"[info] train {len(train_ds)} frames, val {len(val_ds)} frames, device {device}")

    model = CubePoseNet(pretrained=True).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")

    best_pos_cm = math.inf
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, running = time.time(), 0.0
        for img, pos, rot, _ in train_loader:
            img = img.to(device, non_blocking=True)
            pos = pos.to(device, non_blocking=True)
            rot = rot.to(device, non_blocking=True)
            with torch.amp.autocast(device.type, enabled=device.type == "cuda"):
                pred_pos, pred_rot = model(img)
                loss_pos = F.mse_loss(pred_pos, pos)
                loss_rot = F.mse_loss(pred_rot, rot)
                loss = loss_pos + args.rot_loss_weight * loss_rot
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running += loss.item()
        scheduler.step()

        stats = evaluate(model, val_loader, device)
        pos_cm, rot_deg = stats["all"]
        per_phase = "  ".join(
            f"{ph}: {p:.2f}cm/{r:.1f}deg" for ph, (p, r) in stats.items() if ph != "all"
        )
        print(
            f"[epoch {epoch:3d}/{args.epochs}] loss {running / len(train_loader):.5f}  "
            f"val {pos_cm:.2f}cm / {rot_deg:.1f}deg  ({per_phase})  "
            f"[{time.time() - t0:.0f}s]"
        )

        ckpt = {"model": model.state_dict(), "epoch": epoch, "val_stats": stats, "args": vars(args)}
        torch.save(ckpt, os.path.join(args.output_dir, "last.pt"))
        if pos_cm < best_pos_cm:
            best_pos_cm = pos_cm
            torch.save(ckpt, os.path.join(args.output_dir, "best.pt"))
            print(f"[epoch {epoch:3d}] new best val position error: {pos_cm:.2f} cm -> best.pt")

    print(f"[done] best val position error {best_pos_cm:.2f} cm; checkpoints in {args.output_dir}")


if __name__ == "__main__":
    main()
