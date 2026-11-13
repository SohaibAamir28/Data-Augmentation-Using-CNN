#!/usr/bin/env bash
# Reproduces every number and figure in the presentation.
# Run from the project folder:  bash run_all.sh
set -e
PY=${PY:-.venv/Scripts/python}   # or: PY=/path/to/env/python bash run_all.sh
[ -x "$PY" ] || PY=python

# 1) Theory figures (NumPy, from scratch)
"$PY" -s src/demo_transforms.py

# 2) Main experiment: 4 augmentation settings x 3 seeds, 10k training images (low-data regime)
for seed in 0 1 2; do
  for aug in none geo photo geo+photo; do
    "$PY" -s src/train.py --aug $aug --n-train 10000 --epochs 60 --seed $seed
  done
done

# 3) Interpolation ablation: which interpolator to use inside the augmentation warp
for interp in nearest bicubic; do
  "$PY" -s src/train.py --aug geo+photo --interp $interp --n-train 10000 --epochs 60 --seed 0
done

# 4) Does augmentation still help with 4.5x more data?
for aug in none geo+photo; do
  "$PY" -s src/train.py --aug $aug --n-train 45000 --epochs 40 --seed 0
done

# 5) Tables and charts
"$PY" -s src/analyze.py
