#!/usr/bin/env bash
# State-of-the-art augmentation benchmark (full 45k CIFAR-10 training set, 60 epochs, ResNet-9).
# Run from the project folder:  PY=/path/to/python bash run_sota.sh
set -e
PY=${PY:-python}
for seed in 0 1; do
  for m in baseline ours autoaugment randaugment trivialaugment augmix mixup cutmix combined; do
    if [ -f "results/sota/${m}_n45000_s${seed}.json" ]; then echo "skip $m s$seed"; continue; fi
    for try in 1 2 3; do "$PY" -s src/train_sota.py --method $m --seed $seed && break; echo "retry $m s$seed (attempt $try failed)"; sleep 20; done
  done
done
# CIFAR-10-C evaluation of every saved checkpoint (needs data/CIFAR-10-C.tar)
"$PY" -s src/eval_cifar10c.py
