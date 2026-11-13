# Impact of Data Augmentation on CNN Performance and Robustness

**CS867 Computer Vision, Fall 2026 · Submitted to Dr Jameel Malik, Associate Professor, SEECS**

| Student | CMS |
|---|---|
| Sohaib Aamir | 538108 |
| Abdul Rafeh | 450126 |
| Mahmoob Ul Hassan | 624084 |

Topic assigned during Lecture 7 (Geometric Transformations, slides 15–16): show that the image
transformations from the lecture, used as data augmentation, make a CNN more accurate and more robust.

## Contents

| Path | What it is |
|---|---|
| `DataAugmentation_CNN.pptx` | The presentation (35 slides, speaker notes included) |
| `src/geometry.py` | From-scratch NumPy: translation, rotation, Euclidean, similarity, affine, homography; solving affine (3 point pairs, least squares) and homography (4 pairs, DLT); forward warping; inverse warping; nearest / bilinear / bicubic interpolation |
| `src/demo_transforms.py` | Generates the theory figures in `figures/` using only `geometry.py` |
| `src/augment.py` | GPU augmentation: one random homography per image `H = C·P·T·R·Sh·S·F·C⁻¹`, applied by inverse warping (`grid_sample`); photometric jitter (brightness, contrast, saturation, noise, grayscale); test-time shifts |
| `src/train.py` | Trains a ResNet-9 style CNN on CIFAR-10 with a chosen augmentation; logs train/val/test curves; runs the 13-shift robustness suite |
| `src/analyze.py` | Aggregates all runs into `results/summary.md`, `results/summary.json` and figures |
| `src/train_sota.py` | State-of-the-art benchmark: trains the same CNN on full CIFAR-10 with 9 augmentation methods (ours, crop + flip, AutoAugment, RandAugment, TrivialAugment, AugMix + JSD, Mixup, CutMix, combined); reports clean accuracy, flip test-time augmentation, calibration error (ECE) and the 13 shifts |
| `src/eval_cifar10c.py` | Scores every benchmark model on CIFAR-10-C (15 corruptions × 5 severities) |
| `src/analyze_sota.py` | Aggregates the benchmark into `results/sota/summary.md` and `summary.json` |
| `run_sota.sh` | Reproduces the state-of-the-art benchmark |
| `run_all.sh` | Reproduces every figure and number in the presentation |
| `figures/` | All generated figures |
| `results/runs/*.json` | Raw results of every training run (per-epoch curves, final metrics, robustness) |
| `results/summary.md` | Summary table of all runs |
| `results/run_log.txt` | Full training log |
| `results/sota/` | Benchmark results: one JSON per run, `cifar10c.json`, summary table and log |
| `pptx_build/` | Script that builds the .pptx directly from the result files |

## Experimental design

* **Data:** CIFAR-10. Fixed split: 5,000 training images held out for **validation**; the model trains on
  10,000 of the rest (a low-data regime, where overfitting is easy to see). The official 10,000-image
  **test set** is never used during training.
* **Model:** ResNet-9 style CNN (~6.5M parameters), no dropout. SGD + Nesterov, one-cycle LR (max 0.05),
  weight decay 5e-4, batch 256, 60 epochs, gradient clipping 5, bf16 autocast.
* **Settings compared:** none · geometric · photometric · geometric + photometric, 3 seeds each.
* **Robustness suite** on the test set: rotations, translation, zoom, shear, perspective, flip, darkening,
  low contrast, noise, blur (never used in training) and a combined rotate + dark shift.
* **Ablations:** interpolator inside the augmentation warp (nearest / bilinear / bicubic);
  45,000 training images instead of 10,000.

## Headline results (10k training images, mean of 3 seeds)

| Augmentation | Test acc. | Train–test gap | Mean acc. under 13 shifts |
|---|---|---|---|
| None | 78.5% | 21.5 pts | 58.8% |
| Geometric | 83.6% | 15.2 pts | 75.2% |
| Photometric | 77.9% | 22.1 pts | 64.7% |
| Geometric + photometric | 85.4% | 13.2 pts | 81.3% |

With 45,000 training images: 87.7% (no augmentation) → 92.4% (geometric + photometric).
Full table: `results/summary.md`.

## State-of-the-art benchmark (full CIFAR-10, 45k training images, mean of 2 seeds)

Same CNN, optimiser and schedule for every method; only the augmentation changes.
CIFAR-10-C (Hendrycks & Dietterich, ICLR 2019) is the standard robustness benchmark: 15 corruptions × 5 severities.

| Method | Test acc. | ECE ↓ | Our 13 shifts | CIFAR-10-C mCA |
|---|---|---|---|---|
| Crop + flip (standard baseline) | 93.85 | 1.60 | 76.6 | 73.0 |
| **Ours: homography + photometric** | 93.16 | 1.28 | **90.1** | 86.0 |
| AutoAugment (2019) | **94.66** | 0.85 | 84.2 | 81.0 |
| RandAugment (2020) | 94.63 | 0.99 | 83.0 | 81.1 |
| TrivialAugment (2021) | 94.41 | 0.59 | 86.8 | 83.3 |
| AugMix + JSD (2020) | 94.28 | **0.50** | 87.8 | **87.0** |
| Mixup (2018) | 93.55 | 15.80 | 78.7 | 76.6 |
| CutMix (2019) | 94.06 | 9.77 | 74.7 | 70.5 |
| TrivialAugment + Erasing + CutMix/Mixup | 92.94 | 15.87 | 84.2 | 81.3 |

Our pipeline is 2nd of 9 on CIFAR-10-C (behind AugMix), ahead of AutoAugment, RandAugment and TrivialAugment,
at about 1.5 points lower clean accuracy. Its weak spot is the contrast corruption (≈80% vs ≈91% for AutoAugment).
Full table: `results/sota/summary.md`. Model checkpoints (~230 MB) are not included; `run_sota.sh` regenerates them.

## How to run

Requires Python 3.10+ and an NVIDIA GPU (CUDA) for training. The theory figures run on CPU.

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

python src/demo_transforms.py       # theory figures only (seconds, CPU)
bash run_all.sh                     # everything: figures + 16 training runs + summary (~1 h on an RTX 4090)
```

CIFAR-10 (~170 MB) downloads automatically into `data/` on the first run.

State-of-the-art benchmark (~1.5 h on an RTX 4090). Download CIFAR-10-C first
(`https://zenodo.org/records/2535967/files/CIFAR-10-C.tar`, 2.9 GB) into `data/CIFAR-10-C.tar`, then:

```bash
bash run_sota.sh                    # 18 training runs + CIFAR-10-C evaluation
python src/analyze_sota.py          # summary table
```

## Rebuilding the slide deck (optional)

```bash
cd pptx_build
npm install pptxgenjs@3.12.0
NODE_PATH=./node_modules node build_deck.js     # writes ../DataAugmentation_CNN.pptx from ../results
```
