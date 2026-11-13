"""
State-of-the-art augmentation benchmark on CIFAR-10 (full 45k training set).

Methods (all share the same model, optimiser, schedule, epochs and data split):
  baseline        random crop (pad 4) + horizontal flip        standard CIFAR recipe (He et al. 2016)
  ours            our homography + photometric GPU pipeline    this project (src/augment.py)
  autoaugment     baseline + AutoAugment CIFAR-10 policy       Cubuk et al., CVPR 2019
  randaugment     baseline + RandAugment (N=2, M=9)            Cubuk et al., CVPRW 2020
  trivialaugment  baseline + TrivialAugmentWide                Müller & Hutter, ICCV 2021
  augmix          baseline + AugMix + Jensen-Shannon loss      Hendrycks et al., ICLR 2020
  mixup           baseline + Mixup (alpha=1)                   Zhang et al., ICLR 2018
  cutmix          baseline + CutMix (alpha=1)                  Yun et al., ICCV 2019
  combined        baseline + TrivialAugment + RandomErasing + CutMix/Mixup (50/50)

Metrics: clean test accuracy, flip-TTA accuracy, expected calibration error (ECE),
our 13-shift suite, and (later, eval_cifar10c.py) CIFAR-10-C mean corruption accuracy.
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
import torchvision.transforms.v2 as T

sys.path.insert(0, os.path.dirname(__file__))
from augment import Augmenter
from train import CNN, norm, shifts, evaluate

ROOT = os.path.join(os.path.dirname(__file__), "..")
METHODS = ["baseline", "ours", "autoaugment", "randaugment", "trivialaugment", "augmix", "mixup", "cutmix", "combined"]


# ------------------------------------------------------------------ data pipelines
CPU_POLICY = {"autoaugment", "randaugment", "trivialaugment", "augmix", "combined"}


class PolicyBatches(torch.utils.data.Dataset):
    """Each item is one whole augmented BATCH (uint8), built in a DataLoader worker.
    Item k = (epoch k // nb, batch k % nb); the shuffle order is a fixed function of (seed, epoch),
    so one persistent loader covers every epoch without per-sample collate overhead (slow on Windows).
    Crop + flip are applied afterwards on the GPU (gpu_crop_flip); the policies are photometric or
    global geometric ops, so this order is equivalent to the usual crop -> flip -> policy."""
    def __init__(self, x_uint8, y, method, bs, epochs, seed):
        self.x, self.y, self.method, self.bs, self.seed = x_uint8, y, method, bs, seed
        self.nb = len(y) // bs
        self.n = self.nb * epochs
        self.policy = {
            "autoaugment": T.AutoAugment(T.AutoAugmentPolicy.CIFAR10),
            "randaugment": T.RandAugment(num_ops=2, magnitude=9),
            "trivialaugment": T.TrivialAugmentWide(),
            "combined": T.TrivialAugmentWide(),
            # AugMix without the ops that overlap CIFAR-10-C corruptions (all_ops=False), as in the paper
            "augmix": T.AugMix(severity=3, mixture_width=3, all_ops=False),
        }[method]
        self.erase = T.RandomErasing(p=0.25) if method == "combined" else None

    def __len__(self):
        return self.n

    def __getitem__(self, k):
        ep, b = divmod(k, self.nb)
        perm = torch.randperm(len(self.y), generator=torch.Generator().manual_seed(self.seed * 100003 + ep))
        idx = perm[b * self.bs:(b + 1) * self.bs]
        torch.manual_seed(self.seed * 100003 + k)            # reproducible policy draws per batch
        xs = self.x[idx]
        if self.method == "augmix":                          # clean view + two AugMix views (JSD loss)
            out = torch.stack([torch.stack([x, self.policy(x), self.policy(x)]) for x in xs])
        else:
            out = torch.stack([self.policy(x) for x in xs])
            if self.erase is not None:
                out = torch.stack([self.erase(x) for x in out])
        return out, self.y[idx]


def _one_thread(_):
    torch.set_num_threads(1)   # avoid 16 workers x N intra-op threads fighting over the CPU


def gpu_crop_flip(x, views=1):
    """Standard CIFAR augmentation on the GPU: reflect-pad 4, random 32x32 crop, random h-flip.
    With views>1, consecutive groups of `views` images share the same crop and flip."""
    N, C, H, W = x.shape
    B = N // views
    xp = F.pad(x, (4, 4, 4, 4), mode="reflect")
    oy = torch.randint(0, 9, (B,), device=x.device).repeat_interleave(views)
    ox = torch.randint(0, 9, (B,), device=x.device).repeat_interleave(views)
    flip = (torch.rand(B, device=x.device) < 0.5).repeat_interleave(views)
    ar = torch.arange(H, device=x.device)
    iy = oy[:, None] + ar
    ix = ox[:, None] + ar
    ix = torch.where(flip[:, None], ix.flip(1), ix)
    return xp[torch.arange(N, device=x.device)[:, None, None], :, iy[:, :, None], ix[:, None, :]].permute(0, 3, 1, 2)


def jsd_loss(logits_clean, logits_a1, logits_a2):
    p_c, p_1, p_2 = (F.softmax(l, dim=1) for l in (logits_clean, logits_a1, logits_a2))
    m = torch.clamp((p_c + p_1 + p_2) / 3.0, 1e-7, 1).log()
    return (F.kl_div(m, p_c, reduction="batchmean") + F.kl_div(m, p_1, reduction="batchmean")
            + F.kl_div(m, p_2, reduction="batchmean")) / 3.0


# ------------------------------------------------------------------ extra metrics
@torch.no_grad()
def predict(model, X, bs=1000, tta=False):
    model.eval()
    outs = []
    for i in range(0, len(X), bs):
        x = norm(X[i:i + bs]).contiguous(memory_format=torch.channels_last)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            o = model(x).float().softmax(1)
            if tta:
                o = (o + model(torch.flip(x, dims=[3])).float().softmax(1)) / 2
        outs.append(o)
    return torch.cat(outs)


def ece(probs, y, bins=15):
    conf, pred = probs.max(1)
    acc = (pred == y).float()
    edges = torch.linspace(0, 1, bins + 1, device=probs.device)
    e = torch.zeros((), device=probs.device)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.float().mean() * (conf[m].mean() - acc[m].mean()).abs()
    return 100 * e.item()


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=METHODS)
    ap.add_argument("--n-train", type=int, default=45000)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()

    dev = "cuda"
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    torch.backends.cudnn.benchmark = True

    d = os.path.join(ROOT, "data")
    tr = torchvision.datasets.CIFAR10(d, train=True, download=True)
    te = torchvision.datasets.CIFAR10(d, train=False, download=True)
    Xall_u8 = torch.tensor(tr.data).permute(0, 3, 1, 2).contiguous()
    Yall = torch.tensor(tr.targets)
    g = torch.Generator().manual_seed(1234)                    # identical split to train.py
    perm = torch.randperm(50000, generator=g)
    val_idx, tr_idx = perm[:5000], perm[5000:5000 + a.n_train]
    Xval = Xall_u8[val_idx].float().div(255).to(dev); Yval = Yall[val_idx].to(dev)
    Xte = torch.tensor(te.data).permute(0, 3, 1, 2).float().div(255).to(dev)
    Yte = torch.tensor(te.targets).to(dev)
    Xtr_u8, Ytr = Xall_u8[tr_idx], Yall[tr_idx]

    nb = a.n_train // a.bs
    if a.method in CPU_POLICY:       # per-sample policies need CPU workers, one whole batch per item
        ds = PolicyBatches(Xtr_u8, Ytr, a.method, a.bs, a.epochs, a.seed)
        loader = iter(torch.utils.data.DataLoader(ds, batch_size=None, shuffle=False, num_workers=a.workers,
                                                  pin_memory=False, prefetch_factor=4, persistent_workers=True,
                                                  worker_init_fn=_one_thread))
    else:                            # everything else runs entirely on the GPU
        Xtr_gpu, Ytr_gpu = Xtr_u8.to(dev), Ytr.to(dev)

        def gpu_batches():
            for ep in range(a.epochs):
                p = torch.randperm(len(Ytr_gpu), device=dev)
                for b in range(nb):
                    i = p[b * a.bs:(b + 1) * a.bs]
                    yield Xtr_gpu[i], Ytr_gpu[i]
        loader = gpu_batches()
    ours = Augmenter(geometric=True, photometric=True) if a.method == "ours" else None
    mix = {"mixup": T.MixUp(alpha=1.0, num_classes=10), "cutmix": T.CutMix(alpha=1.0, num_classes=10),
           "combined": T.RandomChoice([T.MixUp(alpha=1.0, num_classes=10), T.CutMix(alpha=1.0, num_classes=10)])}.get(a.method)

    model = CNN().to(dev).to(memory_format=torch.channels_last)
    opt = torch.optim.SGD(model.parameters(), lr=a.lr, momentum=0.9, weight_decay=5e-4, nesterov=True)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.epochs * nb, pct_start=0.2)

    hist = {k: [] for k in ["val_loss", "val_acc", "test_acc"]}
    t0 = time.time()
    for ep in range(a.epochs):
        model.train()
        for _ in range(nb):
            xb, yb = next(loader)
            xb = xb.to(dev).float().div(255)          # blocking copy: async copies from pinned worker buffers crashed on Windows
            yb = yb.to(dev)
            if ours is not None:
                xb = ours(xb)                                   # our pipeline replaces crop/flip
            elif a.method == "augmix":
                B = xb.shape[0]
                xb = gpu_crop_flip(xb.flatten(0, 1), views=3).view(B, 3, *xb.shape[2:])
            else:
                xb = gpu_crop_flip(xb)
            if a.method == "augmix":
                x3 = norm(xb.flatten(0, 1)).contiguous(memory_format=torch.channels_last)   # (3B,C,H,W)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    lo = model(x3).float().view(B, 3, -1)
                loss = F.cross_entropy(lo[:, 0], yb) + 12.0 * jsd_loss(lo[:, 0], lo[:, 1], lo[:, 2])
            else:
                target = yb
                if mix is not None:
                    xb, target = mix(xb, yb)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    out = model(norm(xb).contiguous(memory_format=torch.channels_last))
                loss = F.cross_entropy(out.float(), target)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step(); sched.step()
        vl, va = evaluate(model, Xval, Yval)
        _, ta = evaluate(model, Xte, Yte)
        for k, v in zip(hist, [vl, va, ta]):
            hist[k].append(round(v, 4))
        print(f"[{a.method} s={a.seed}] ep {ep + 1:3d}  val {vl:.3f}/{va:.2f}%  test {ta:.2f}%  ({time.time() - t0:.0f}s)", flush=True)

    probs = predict(model, Xte)
    probs_tta = predict(model, Xte, tta=True)
    robust = {name: round(evaluate(model, Xte, Yte, fn)[1], 2) for name, fn in shifts(dev).items()}
    res = dict(vars(a), history=hist, robustness=robust, seconds=round(time.time() - t0, 1),
               final=dict(val_acc=hist["val_acc"][-1], val_loss=hist["val_loss"][-1],
                          test_acc=round(100 * (probs.argmax(1) == Yte).float().mean().item(), 2),
                          test_acc_tta=round(100 * (probs_tta.argmax(1) == Yte).float().mean().item(), 2),
                          ece=round(ece(probs, Yte), 2),
                          shift_mean=round(float(np.mean([v for k, v in robust.items() if k != "Clean"])), 2)))
    out = os.path.join(ROOT, "results", "sota")
    os.makedirs(os.path.join(out, "checkpoints"), exist_ok=True)
    name = f"{a.method}_n{a.n_train}_s{a.seed}"
    torch.save({k: (v.half() if v.is_floating_point() else v) for k, v in model.state_dict().items()}, os.path.join(out, "checkpoints", name + ".pt"))
    with open(os.path.join(out, name + ".json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1, ensure_ascii=False)
    print("final:", json.dumps(res["final"]))


if __name__ == "__main__":
    main()
