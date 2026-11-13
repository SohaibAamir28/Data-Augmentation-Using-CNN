"""
CNN experiment: does data augmentation reduce overfitting and make inference more robust?

Data (CIFAR-10):
  train      : N images drawn from the 50k training set (default 10k -> low-data regime)
  validation : 5k held-out training images (never trained on) -> validation loss/accuracy
  test       : the official 10k test set (unseen) -> clean accuracy + robustness suite

Usage:
  python src/train.py --aug none        --n-train 10000 --seed 0
  python src/train.py --aug geo         ...
  python src/train.py --aug photo       ...
  python src/train.py --aug geo+photo   --interp bicubic ...
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

sys.path.insert(0, os.path.dirname(__file__))
from augment import Augmenter, fixed_homography, inverse_warp, adjust, gaussian_blur

ROOT = os.path.join(os.path.dirname(__file__), "..")
MEAN = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
STD = torch.tensor([0.2470, 0.2435, 0.2616]).view(1, 3, 1, 1)


# ------------------------------------------------------------------ model
def block(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1, bias=False),
                         nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class Residual(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.f = nn.Sequential(block(c, c), block(c, c))

    def forward(self, x):
        return x + self.f(x)


class CNN(nn.Module):
    """ResNet-9 style CNN (~6.5M params). No dropout, so overfitting is visible without augmentation."""
    def __init__(self, n_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            block(3, 64), block(64, 128), nn.MaxPool2d(2), Residual(128),
            block(128, 256), nn.MaxPool2d(2),
            block(256, 512), nn.MaxPool2d(2), Residual(512),
            nn.AdaptiveMaxPool2d(1), nn.Flatten(), nn.Linear(512, n_classes))

    def forward(self, x):
        return self.net(x) * 0.125   # output scaling (standard ResNet-9 trick) keeps early training stable


# ------------------------------------------------------------------ data
def load_cifar(dev):
    d = os.path.join(ROOT, "data")
    tr = torchvision.datasets.CIFAR10(d, train=True, download=True)
    te = torchvision.datasets.CIFAR10(d, train=False, download=True)
    to_t = lambda a: torch.tensor(a).permute(0, 3, 1, 2).float().div(255).to(dev)
    return (to_t(tr.data), torch.tensor(tr.targets, device=dev),
            to_t(te.data), torch.tensor(te.targets, device=dev))


def norm(x):
    return (x - MEAN.to(x.device)) / STD.to(x.device)


# ------------------------------------------------------------------ robustness suite
def shifts(dev):
    """Test-time distribution shifts. Each maps a (B,3,32,32) batch in [0,1] to a shifted batch."""
    H = lambda **k: (lambda x: inverse_warp(x, fixed_homography(x.shape[0], 32, dev, **k), "bilinear"))
    return {
        "Clean": lambda x: x,
        "Rotate 15°": H(rot=15),
        "Rotate 30°": H(rot=30),
        "Translate 4px": H(tx=4, ty=4),
        "Zoom out 0.8x": H(scale=0.8),
        "Zoom in 1.2x": H(scale=1.2),
        "Shear 15° (affine)": H(shear=15),
        "Perspective (homography)": H(persp=(0.006, 0.004)),
        "Horizontal flip": H(flip=True),
        "Dark (brightness 0.5)": lambda x: adjust(x, brightness=0.5),
        "Low contrast 0.4": lambda x: adjust(x, contrast=0.4),
        "Gaussian noise σ=0.08": lambda x: (x + 0.08 * torch.randn(x.shape, device=x.device,
                                             generator=torch.Generator(x.device).manual_seed(0))).clamp(0, 1),
        "Blur σ=1 (never trained on)": lambda x: gaussian_blur(x, 1.0),
        "Rotate 15° + dark": lambda x: adjust(H(rot=15)(x), brightness=0.5),
    }


@torch.no_grad()
def evaluate(model, X, Y, fn=lambda x: x, bs=1000):
    model.eval()
    loss, correct = 0.0, 0
    for i in range(0, len(X), bs):
        x = norm(fn(X[i:i + bs])).contiguous(memory_format=torch.channels_last)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(x)
        loss += F.cross_entropy(out.float(), Y[i:i + bs], reduction="sum").item()
        correct += (out.argmax(1) == Y[i:i + bs]).sum().item()
    return loss / len(X), 100.0 * correct / len(X)


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aug", default="none", choices=["none", "geo", "photo", "geo+photo"])
    ap.add_argument("--interp", default="bilinear", choices=["nearest", "bilinear", "bicubic"])
    ap.add_argument("--n-train", type=int, default=10000)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    dev = "cuda"
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    torch.backends.cudnn.benchmark = True
    Xall, Yall, Xte, Yte = load_cifar(dev)

    # fixed split (same for every run): last 5k of the training set = validation
    g = torch.Generator().manual_seed(1234)
    perm = torch.randperm(50000, generator=g).to(dev)
    val_idx, pool = perm[:5000], perm[5000:]
    tr_idx = pool[:a.n_train]
    Xtr, Ytr, Xval, Yval = Xall[tr_idx], Yall[tr_idx], Xall[val_idx], Yall[val_idx]

    model = CNN().to(dev).to(memory_format=torch.channels_last)
    opt = torch.optim.SGD(model.parameters(), lr=a.lr, momentum=0.9, weight_decay=5e-4, nesterov=True)
    steps_per_epoch = (len(Xtr) + a.bs - 1) // a.bs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.epochs * steps_per_epoch,
                                                pct_start=0.2)
    aug = Augmenter(geometric="geo" in a.aug, photometric="photo" in a.aug, interp=a.interp)

    hist = {k: [] for k in ["train_loss", "train_acc", "val_loss", "val_acc", "test_loss", "test_acc"]}
    t0 = time.time()
    for ep in range(a.epochs):
        model.train()
        idx = torch.randperm(len(Xtr), device=dev)
        for i in range(0, len(Xtr), a.bs):
            b = idx[i:i + a.bs]
            x = norm(aug(Xtr[b])).contiguous(memory_format=torch.channels_last)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = F.cross_entropy(model(x), Ytr[b])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step(); sched.step()
        # train metrics measured on the clean (un-augmented) training images
        trl, tra = evaluate(model, Xtr, Ytr)
        vl, va = evaluate(model, Xval, Yval)
        tel, tea = evaluate(model, Xte, Yte)
        for k, v in zip(hist, [trl, tra, vl, va, tel, tea]):
            hist[k].append(round(v, 4))
        print(f"[{a.aug}/{a.interp} n={a.n_train} s={a.seed}] ep {ep + 1:3d}  "
              f"train {trl:.3f}/{tra:.1f}%  val {vl:.3f}/{va:.1f}%  test {tea:.1f}%  ({time.time() - t0:.0f}s)",
              flush=True)

    robust = {name: round(evaluate(model, Xte, Yte, fn)[1], 2) for name, fn in shifts(dev).items()}
    res = dict(vars(a), history=hist, robustness=robust, seconds=round(time.time() - t0, 1),
               final=dict(train_acc=hist["train_acc"][-1], val_acc=hist["val_acc"][-1],
                          val_loss=hist["val_loss"][-1], best_val_loss=min(hist["val_loss"]),
                          test_acc=hist["test_acc"][-1],
                          gap=round(hist["train_acc"][-1] - hist["test_acc"][-1], 2)))
    out_dir = os.path.join(ROOT, "results", "runs")
    os.makedirs(out_dir, exist_ok=True)
    name = f"{a.aug.replace('+', '_')}_{a.interp}_n{a.n_train}_s{a.seed}{a.tag}.json"
    with open(os.path.join(out_dir, name), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1, ensure_ascii=False)
    print("robustness:", json.dumps(robust))


if __name__ == "__main__":
    main()
