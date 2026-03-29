# Model Evaluation Comparison

## Date: 2026-03-29

## Models

| Model | Checkpoint | Training Config |
|---|---|---|
| **Pretrained (repo)** | `pretrained_models/reach/policy.pt` | 4096 envs, 1000 iters, ~98M steps |
| **Ours** | `logs/rsl_rl/reach_gen3/2026-03-29_11-01-50/model_1999.pt` | 2048 envs, 2000 iters, ~98M steps |

## Evaluation Setup

- **Task:** Gen3-Reach-v0
- **Eval envs:** 64
- **Eval steps:** 3000
- **Mode:** headless

## Results

| Metric | Pretrained | Ours | Winner |
|---|---|---|---|
| Mean reward | -0.0004 | -0.0002 | Ours |
| Mean position error | 0.0767 m | 0.0558 m | Ours |
| Mean orientation error | 0.5447 rad | 0.6180 rad | Pretrained |
| Std position error | 0.0381 m | 0.0371 m | Ours |
| Std orientation error | 0.3892 rad | 0.4162 rad | Pretrained |

## Notes

- Both models were trained on ~98M total steps but with different configurations:
  - Pretrained: 4096 envs x 24 steps x 1000 iterations (larger batch per update)
  - Ours: 2048 envs x 24 steps x 2000 iterations (smaller batch, more updates)
- Our model achieves better position accuracy (5.6cm vs 7.7cm)
- Pretrained model achieves better orientation accuracy (0.54 vs 0.62 rad)
- The pretrained `policy.pt` is a TorchScript JIT export; ours is a standard rsl-rl checkpoint
- Training time for our model: ~8 min 44 sec on RTX 4070 Laptop (8GB VRAM)

## How to Reproduce

```bash
# Evaluate pretrained
python scripts/rsl_rl/eval/eval.py --task Gen3-Reach-v0 --checkpoint pretrained_models/reach/policy.pt --headless

# Evaluate ours
python scripts/rsl_rl/eval/eval.py --task Gen3-Reach-v0 --headless
```
