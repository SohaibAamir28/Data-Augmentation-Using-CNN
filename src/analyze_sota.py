"""Aggregate results/sota/*.json (+ cifar10c.json) into results/sota/summary.{json,md}."""
import glob
import json
import os
from collections import defaultdict

import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..")
D = os.path.join(ROOT, "results", "sota")
ORDER = ["baseline", "ours", "autoaugment", "randaugment", "trivialaugment", "augmix", "mixup", "cutmix", "combined"]
LABEL = {"baseline": "Crop + flip (standard baseline)", "ours": "Ours: homography + photometric",
         "autoaugment": "AutoAugment (2019)", "randaugment": "RandAugment (2020)",
         "trivialaugment": "TrivialAugment (2021)", "augmix": "AugMix + JSD (2020)", "mixup": "Mixup (2018)",
         "cutmix": "CutMix (2019)", "combined": "TrivialAugment + Erasing + CutMix/Mixup"}


def main():
    runs = defaultdict(list)
    for p in glob.glob(os.path.join(D, "*_n45000_s*.json")):
        r = json.load(open(p, encoding="utf-8"))
        runs[r["method"]].append(r)
    c10c = json.load(open(os.path.join(D, "cifar10c.json"))) if os.path.exists(os.path.join(D, "cifar10c.json")) else {}
    rows = []
    for m in ORDER:
        rs = runs.get(m)
        if not rs:
            continue
        f = lambda k: [r["final"][k] for r in rs]
        cc = [c10c[f"{m}_n45000_s{r['seed']}"] for r in rs if f"{m}_n45000_s{r['seed']}" in c10c]
        row = dict(method=m, label=LABEL[m], seeds=len(rs))
        for k in ["test_acc", "test_acc_tta", "ece", "shift_mean", "val_loss"]:
            row[k], row[k + "_std"] = round(float(np.mean(f(k))), 2), round(float(np.std(f(k))), 2)
        if cc:
            row["mCA"] = round(float(np.mean([c["mCA"] for c in cc])), 2)
            row["mCA_std"] = round(float(np.std([c["mCA"] for c in cc])), 2)
            row["by_group"] = {g: round(float(np.mean([c["by_group"][g] for c in cc])), 2) for g in cc[0]["by_group"]}
            row["by_severity"] = [round(float(np.mean([c["by_severity"][s] for c in cc])), 2) for s in range(5)]
            row["c10c_seeds"] = len(cc)
        row["minutes"] = round(float(np.mean([r["seconds"] for r in rs])) / 60, 1)
        rows.append(row)
    json.dump(rows, open(os.path.join(D, "summary.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    lines = ["| Method | Seeds | Test acc. | + flip TTA | ECE ↓ | Our 13 shifts | CIFAR-10-C mCA | mCE ↓ | min/run |", "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        mca = f"{r['mCA']:.1f}" if "mCA" in r else "–"
        mce = f"{100 - r['mCA']:.1f}" if "mCA" in r else "–"
        lines.append(f"| {r['label']} | {r['seeds']} | {r['test_acc']:.2f} ± {r['test_acc_std']:.2f} | {r['test_acc_tta']:.2f} | "
                     f"{r['ece']:.2f} | {r['shift_mean']:.1f} | {mca} | {mce} | {r['minutes']} |")
    open(os.path.join(D, "summary.md"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
