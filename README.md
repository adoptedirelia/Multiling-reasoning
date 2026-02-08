# Multiling-reasoning


## Evaluation

```bash
python -m src.eval.main --config ./configs/eval_config_example.json
```

## SFT training

```bash
torchrun --nproc_per_node=1 train.py --config ./configs/train_config_example.json
```

## RL training

