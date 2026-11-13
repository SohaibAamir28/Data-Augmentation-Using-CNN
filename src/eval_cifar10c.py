"""
Evaluate every checkpoint in results/sota/checkpoints on CIFAR-10-C (Hendrycks & Dietterich, ICLR 2019).

Standard protocol: 15 corruptions x 5 severities x 10,000 test images.
Reported: accuracy per corruption/severity, mean corruption accuracy (mCA) and mean corruption error
mCE = 100 - mCA (unnormalised, as used for CIFAR in the AugMix paper).
Reads the .npy files directly from data/CIFAR-10-C.tar (no extraction).
"""
import glob
import io
import json
import os
import sys
import tarfile

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from train import CNN, norm

ROOT = os.path.join(os.path.dirname(__file__), "..")
TAR = os.path.join(ROOT, "data", "CIFAR-10-C.tar")
CORRUPTIONS = {
    "noise": ["gaussian_noise", "shot_noise", "impulse_noise"],
    "blur": ["defocus_blur", "glass_blur", "motion_blur", "zoom_blur"],
    "weather": ["snow", "frost", "fog", "brightness"],
    "digital": ["contrast", "elastic_transform", "pixelate", "jpeg_compression"],
}
ALL = [c for g in CORRUPTIONS.values() for c in g]


def load_tar():
    try:
        tf = tarfile.open(TAR)
        members = {os.path.basename(m.name): m for m in tf.getmembers()}
        need = [c + ".npy" for c in ALL] + ["labels.npy"]
        missing = [n for n in need if n not in members]
        if missing:
            raise IOError(f"missing {missing}")
        return tf, members
    except Exception as e:
        print(f"CIFAR-10-C not available yet ({type(e).__name__}: {e}); skipping.")
        sys.exit(0)


@torch.no_grad()
def accuracy(model, X_u8, Y, bs=2000):
    correct = 0
    for i in range(0, len(X_u8), bs):
        x = torch.from_numpy(X_u8[i:i + bs]).cuda().permute(0, 3, 1, 2).float().div(255)
        x = norm(x).contiguous(memory_format=torch.channels_last)
        correct += (model(x).argmax(1) == Y[i:i + bs]).sum().item()   # fp32: bf16 path crashed on this GPU
    return 100.0 * correct / len(X_u8)


def main():
    tf, members = load_tar()
    ckpts = sorted(glob.glob(os.path.join(ROOT, "results", "sota", "checkpoints", "*.pt")))
    out_path = os.path.join(ROOT, "results", "sota", "cifar10c.json")
    part_path = os.path.join(ROOT, "results", "sota", "cifar10c_partial.json")   # saved after every corruption
    results = json.load(open(out_path)) if os.path.exists(out_path) else {}
    per = json.load(open(part_path)) if os.path.exists(part_path) else {}
    names = [os.path.basename(c)[:-3] for c in ckpts if os.path.basename(c)[:-3] not in results]
    if not names:
        print("nothing to evaluate"); return
    models = {}
    for n in names:
        per.setdefault(n, {})
        m = CNN().cuda()
        sd = torch.load(os.path.join(ROOT, "results", "sota", "checkpoints", n + ".pt"))
        m.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in sd.items()})
        models[n] = m.eval().to(memory_format=torch.channels_last)

    labels = torch.from_numpy(np.load(io.BytesIO(tf.extractfile(members["labels.npy"]).read()))).long().cuda()
    for corr in ALL:                                   # load each corruption once, score every model
        todo = [n for n in models if corr not in per[n]]
        if not todo:
            continue
        X = np.load(io.BytesIO(tf.extractfile(members[corr + ".npy"]).read()))
        for n in todo:
            per[n][corr] = [round(accuracy(models[n], X[s * 10000:(s + 1) * 10000], labels[s * 10000:(s + 1) * 10000]), 2)
                            for s in range(5)]
        json.dump(per, open(part_path, "w"), indent=1)
        print(corr, {n: round(float(np.mean(per[n][corr])), 1) for n in todo}, flush=True)
    for name in models:
        d = per[name]
        mca = float(np.mean([np.mean(d[c]) for c in ALL]))
        results[name] = {"per_corruption": d, "mCA": round(mca, 2), "mCE": round(100 - mca, 2),
                         "by_group": {g: round(float(np.mean([np.mean(d[c]) for c in cs])), 2) for g, cs in CORRUPTIONS.items()},
                         "by_severity": [round(float(np.mean([d[c][s] for c in ALL])), 2) for s in range(5)]}
    json.dump(results, open(out_path, "w"), indent=1)
    print("saved", out_path)


if __name__ == "__main__":
    main()
