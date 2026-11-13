"""Aggregate results/runs/*.json into figures/ and results/summary.{json,md}."""
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from augment import Augmenter

ROOT = os.path.join(os.path.dirname(__file__), "..")
FIG = os.path.join(ROOT, "figures")
os.makedirs(FIG, exist_ok=True)

INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3de", "#fbfbf8"
COL = {"none": "#2a78d6", "geo": "#eb6834", "photo": "#1baf7a", "geo+photo": "#eda100"}
COL_DARK_TEXT = "#6b4a00"
LABEL = {"none": "No augmentation", "geo": "Geometric", "photo": "Photometric", "geo+photo": "Geometric + photometric"}
ORDER = ["none", "geo", "photo", "geo+photo"]
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 15, "text.color": INK, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.edgecolor": GRID, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 1, "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF, "lines.linewidth": 2.4,
    "legend.frameon": False,
})


def load():
    runs = [json.load(open(p, encoding="utf-8")) for p in glob.glob(os.path.join(ROOT, "results", "runs", "*.json"))]
    groups = defaultdict(list)
    for r in runs:
        groups[(r["aug"], r["interp"], r["n_train"])].append(r)
    return groups


def mean_std(rs, f):
    v = np.array([f(r) for r in rs], float)
    return float(v.mean()), float(v.std()), len(v)


def fig_curves(groups, n):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.2))
    ax = axes[0]
    for aug in ["none", "geo+photo"]:
        rs = groups.get((aug, "bilinear", n))
        if not rs:
            continue
        tr = np.mean([r["history"]["train_acc"] for r in rs], 0)
        va = np.mean([r["history"]["val_acc"] for r in rs], 0)
        ep = np.arange(1, len(tr) + 1)
        ax.plot(ep, tr, color=COL[aug], ls="--")
        ax.plot(ep, va, color=COL[aug])
        ax.text(ep[-1] + 1, tr[-1], f"{LABEL[aug]} · train", color=INK2, va="center", fontsize=13)
        ax.text(ep[-1] + 1, va[-1], f"{LABEL[aug]} · val", color=INK, va="center", fontsize=13)
    ax.set_xlim(0, len(ep) * 1.55)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy (%)")
    ax.set_title("Accuracy: train (dashed) vs validation (solid)", loc="left", fontsize=16)

    ax = axes[1]
    for aug in ORDER:
        rs = groups.get((aug, "bilinear", n))
        if not rs:
            continue
        vl = np.mean([r["history"]["val_loss"] for r in rs], 0)
        ax.plot(np.arange(1, len(vl) + 1), vl, color=COL[aug], label=LABEL[aug])
    ax.set_xlabel("Epoch"); ax.set_ylabel("Validation loss (cross-entropy)")
    ax.set_title("Validation loss", loc="left", fontsize=16)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_curves.png"), dpi=150)
    plt.close(fig)


def fig_robustness(groups, n):
    a, b = groups.get(("none", "bilinear", n)), groups.get(("geo+photo", "bilinear", n))
    if not a or not b:
        return
    names = list(a[0]["robustness"].keys())
    ma = [np.mean([r["robustness"][k] for r in a]) for k in names]
    mb = [np.mean([r["robustness"][k] for r in b]) for k in names]
    y = np.arange(len(names))[::-1]
    fig, ax = plt.subplots(figsize=(14, 8.4))
    h = 0.38
    ax.barh(y + h / 2, ma, height=h - 0.04, color=COL["none"], label=LABEL["none"])
    ax.barh(y - h / 2, mb, height=h - 0.04, color=COL["geo+photo"], label=LABEL["geo+photo"])
    for yi, va, vb in zip(y, ma, mb):
        ax.text(vb + 0.8, yi - h / 2, f"{vb:.1f}  (+{vb - va:.1f})", va="center", fontsize=12, color=INK)
        ax.text(va + 0.8, yi + h / 2, f"{va:.1f}", va="center", fontsize=12, color=INK2)
    ax.set_yticks(y); ax.set_yticklabels(names)
    ax.set_xlim(0, 105); ax.set_xlabel("Test accuracy (%) on the 10k unseen test images")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right", ncol=2, bbox_to_anchor=(1, 1.0))
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_robustness.png"), dpi=150)
    plt.close(fig)


def fig_aug_examples():
    import torchvision
    ds = torchvision.datasets.CIFAR10(os.path.join(ROOT, "data"), train=False, download=False)
    idx = [3, 10, 21, 30, 44, 61]
    x = torch.tensor(ds.data[idx]).permute(0, 3, 1, 2).float().div(255).cuda()
    torch.manual_seed(7)
    rows = [("Original", x),
            ("Geometric", Augmenter(True, False)(x)),
            ("Photometric", Augmenter(False, True)(x)),
            ("Both", Augmenter(True, True)(x))]
    fig, axes = plt.subplots(len(rows), len(idx), figsize=(12, 8.4))
    for r, (name, im) in enumerate(rows):
        for c in range(len(idx)):
            ax = axes[r, c]
            ax.imshow(im[c].permute(1, 2, 0).cpu().numpy(), interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
            for s in ax.spines.values():
                s.set_visible(False)
            if c == 0:
                ax.set_ylabel(name, fontsize=16, color=INK, rotation=0, ha="right", va="center")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_aug_examples.png"), dpi=150)
    plt.close(fig)


def main():
    groups = load()
    summary = []
    for (aug, interp, n), rs in sorted(groups.items(), key=lambda kv: (kv[0][2], ORDER.index(kv[0][0]), kv[0][1])):
        row = dict(aug=aug, interp=interp, n_train=n, seeds=len(rs))
        for key in ["train_acc", "val_acc", "best_val_loss", "val_loss", "test_acc", "gap"]:
            m, s, _ = mean_std(rs, lambda r: r["final"][key])
            row[key], row[key + "_std"] = round(m, 2), round(s, 2)
        rob = [k for k in rs[0]["robustness"] if k != "Clean"]
        row["shifted_mean_acc"] = round(float(np.mean([[r["robustness"][k] for k in rob] for r in rs])), 2)
        row["robustness"] = {k: round(float(np.mean([r["robustness"][k] for r in rs])), 2) for k in rs[0]["robustness"]}
        row["minutes"] = round(float(np.mean([r["seconds"] for r in rs])) / 60, 1)
        summary.append(row)
    json.dump(summary, open(os.path.join(ROOT, "results", "summary.json"), "w", encoding="utf-8"),
              indent=1, ensure_ascii=False)

    lines = ["| Aug | Interp | N train | Seeds | Train acc | Val acc | Val loss (final / best) | Test acc | Train-test gap | Mean acc under 13 shifts |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for r in summary:
        lines.append(f"| {r['aug']} | {r['interp']} | {r['n_train']} | {r['seeds']} | {r['train_acc']:.1f} | "
                     f"{r['val_acc']:.1f} | {r['val_loss']:.3f} / {r['best_val_loss']:.3f} | "
                     f"{r['test_acc']:.1f} ± {r['test_acc_std']:.1f} | {r['gap']:.1f} | {r['shifted_mean_acc']:.1f} |")
    open(os.path.join(ROOT, "results", "summary.md"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("\n".join(lines))

    n_main = 10000
    fig_curves(groups, n_main)
    fig_robustness(groups, n_main)
    fig_aug_examples()


if __name__ == "__main__":
    main()
