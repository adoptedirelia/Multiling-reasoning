# Multiling-reasoning


## Evaluation

```bash
python -m src.eval.main --config ./configs/eval_config_example.json
```

## SFT training

```bash
torchrun --nproc_per_node=1 -m trainer --config /home/dzhang98/code/Multiling-reasoning/configs/train_config_example.json
```

## RL training

