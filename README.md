# Multiling-reasoning


## Evaluation

```bash
python -m src.eval.main
```

## SFT training

```bash
torchrun --nproc_per_node=1 train.py --config ./configs/train_config_example.json
```

## RL training

- to be done