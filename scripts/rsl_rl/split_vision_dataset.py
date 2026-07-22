"""Split vision_dataset/labels.jsonl into train/val/test, randomly shuffled.

Frames are collected in rollout order, so the file front-loads REACH-phase
("cube still on table") frames -- every env starts each episode in REACH at
the same time, and only reaches GRASP/CARRY later. A sequential split would
give each split a different, unrepresentative phase mix. This shuffles first
(fixed seed, for reproducibility) so train/val/test each get a representative
sample of every phase.

Images are left untouched in images/; each split file just references the
same "file" paths as labels.jsonl.
"""

import argparse
import json
import random
from collections import Counter

parser = argparse.ArgumentParser(description="Split labels.jsonl into train/val/test.")
parser.add_argument("--dataset_dir", type=str, default="vision_dataset")
parser.add_argument("--num_val", type=int, default=1000)
parser.add_argument("--num_test", type=int, default=1000)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

labels_path = f"{args.dataset_dir}/labels.jsonl"
with open(labels_path) as f:
    records = [json.loads(l) for l in f]

random.seed(args.seed)
random.shuffle(records)

test_records = records[: args.num_test]
val_records = records[args.num_test : args.num_test + args.num_val]
train_records = records[args.num_test + args.num_val :]

splits = {"train": train_records, "val": val_records, "test": test_records}
for name, split_records in splits.items():
    out_path = f"{args.dataset_dir}/{name}_labels.jsonl"
    with open(out_path, "w") as f:
        for r in split_records:
            f.write(json.dumps(r) + "\n")
    phase_counts = Counter(r["phase"] for r in split_records)
    print(f"{name}: {len(split_records)} frames -> {out_path}")
    print(f"  phase distribution: {dict(phase_counts)}")
