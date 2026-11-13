"""
Generates the theory figures for the presentation using ONLY src/geometry.py
(our from-scratch implementation):

  fig_transform_family.png   translation / rotation / Euclidean / similarity / affine / homography
  fig_forward_vs_inverse.png forward warping leaves holes, inverse warping does not
  fig_interpolation.png      nearest vs bilinear vs bicubic (zoomed crop)
  fig_point_pairs.png        solving affine (3 pairs) and homography (4 pairs) from correspondences
  interpolation_error.json   round-trip error of each interpolator (rotate +θ then -θ)
"""
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cbook
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
import geometry as G

OUT = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(OUT, exist_ok=True)

INK = "#0b0b0b"
INK2 = "#52514e"
BLUE = "#2a78d6"
ORANGE = "#eb6834"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 13, "text.color": INK,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "figure.facecolor": "#fbfbf8", "savefig.facecolor": "#fbfbf8",
})


def load_image(size=256):
    with cbook.get_sample_data("grace_hopper.jpg") as f:
        im = Image.open(f).convert("RGB")
    w, h = im.size
    s = min(w, h)
    im = im.crop(((w - s) // 2, 0, (w - s) // 2 + s, s)).resize((size, size), Image.LANCZOS)
    return np.asarray(im).astype(np.float64)


def show(ax, img, title):
    ax.imshow(np.clip(img, 0, 255).astype(np.uint8), interpolation="nearest")
    ax.set_title(title, fontsize=14, color=INK)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color("#d9d8d3")


def fig_transform_family(img):
    n = img.shape[0]
    T = {
        "Original\n(identity, 0 DoF)": np.eye(3),
        "Translation\n2 DoF": G.translation(40, 25),
        "Rotation\n1 DoF": G.about_center(G.rotation(25), n, n),
        "Euclidean (rigid)\n3 DoF": G.about_center(G.euclidean(-20, 30, -15), n, n),
        "Similarity\n4 DoF": G.about_center(G.similarity(0.75, 30, 0, 0), n, n),
        "Affine\n6 DoF": G.about_center(G.affine(0.9, 0.35, 0, 0.1, 0.8, 0), n, n),
        "Homography\n8 DoF": G.solve_homography(
            [[0, 0], [n - 1, 0], [n - 1, n - 1], [0, n - 1]],
            [[30, 10], [n - 50, 40], [n - 20, n - 30], [10, n - 5]]),
    }
    fig, axes = plt.subplots(1, 7, figsize=(21, 3.9))
    for ax, (name, M) in zip(axes, T.items()):
        show(ax, G.inverse_warp(img, M, method="bilinear"), name)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_transform_family.png"), dpi=150, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


def fig_forward_vs_inverse(img):
    n = img.shape[0]
    M = G.about_center(G.similarity(1.35, 20, 0, 0), n, n)  # zoom-in + rotate: worst case for holes
    fw, holes = G.forward_warp(img, M)
    iw = G.inverse_warp(img, M, method="bilinear")
    inside = G.inverse_warp(np.ones((n, n)), M, method="nearest") > 0.5
    hole_pct = 100 * (holes & inside).sum() / inside.sum()

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.9))
    show(axes[0], img, "Source image I")
    show(axes[1], fw, f"Forward warping\n{hole_pct:.0f}% of target pixels are holes")
    show(axes[2], iw, "Inverse warping + bilinear\n0% holes")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_forward_vs_inverse.png"), dpi=150, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    return hole_pct


def fig_interpolation(img):
    n = img.shape[0]
    crop = img[88:112, 104:128]   # 24x24 detail around the eye / glasses
    # 10.7x zoom + 10° rotation about the crop centre, in ONE inverse warp
    M = G.translation(127.5, 127.5) @ G.similarity(256 / 24, 10, 0, 0) @ G.translation(-11.5, -11.5)
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 5.2))
    for ax, m, label in zip(axes, ["nearest", "bilinear", "bicubic"],
                            ["Nearest neighbour\n1 pixel: blocky", "Bilinear\n4 pixels: smooth, a bit soft",
                             "Bicubic\n16 pixels: sharper edges"]):
        big = G.inverse_warp(crop, M, out_shape=(256, 256), method=m)
        show(ax, big, label)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_interpolation.png"), dpi=150, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)

    # round-trip error: rotate by +θ and back by -θ, compare to original centre region
    errs = {}
    for m in ["nearest", "bilinear", "bicubic"]:
        e = []
        for th in [5, 10, 15, 20, 30]:
            R = G.about_center(G.rotation(th), n, n)
            back = G.inverse_warp(G.inverse_warp(img, R, method=m), np.linalg.inv(R), method=m)
            c = slice(64, 192)
            e.append(np.sqrt(np.mean((back[c, c] - img[c, c]) ** 2)))
        errs[m] = round(float(np.mean(e)), 2)
    return errs


def fig_point_pairs(img):
    n = img.shape[0]
    src3 = np.array([[40, 40], [210, 50], [60, 220]])
    dst3 = np.array([[60, 30], [220, 90], [30, 200]])
    A = G.solve_affine(src3, dst3)
    src4 = np.array([[0, 0], [n - 1, 0], [n - 1, n - 1], [0, n - 1]])
    dst4 = np.array([[40, 20], [n - 30, 50], [n - 10, n - 20], [5, n - 40]])
    H = G.solve_homography(src4, dst4)

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.9))
    show(axes[0], img, "Source + 3 points")
    axes[0].scatter(src3[:, 0], src3[:, 1], s=110, c=ORANGE, edgecolors="white", linewidths=2, zorder=3)
    show(axes[1], G.inverse_warp(img, A), "Affine solved from 3 pairs")
    axes[1].scatter(dst3[:, 0], dst3[:, 1], s=110, c=ORANGE, edgecolors="white", linewidths=2, zorder=3)
    show(axes[2], img, "Source + 4 corners")
    axes[2].scatter(src4[:, 0], src4[:, 1], s=110, c=BLUE, edgecolors="white", linewidths=2, zorder=3, clip_on=False)
    show(axes[3], G.inverse_warp(img, H), "Homography solved from 4 pairs")
    axes[3].scatter(dst4[:, 0], dst4[:, 1], s=110, c=BLUE, edgecolors="white", linewidths=2, zorder=3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_point_pairs.png"), dpi=150, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    # sanity: solved transforms reproduce the target points
    return (float(np.abs(G.apply_to_points(A, src3) - dst3).max()),
            float(np.abs(G.apply_to_points(H, src4) - dst4).max()))


if __name__ == "__main__":
    img = load_image()
    fig_transform_family(img)
    holes = fig_forward_vs_inverse(img)
    errs = fig_interpolation(img)
    fit_err = fig_point_pairs(img)
    info = {"forward_warp_hole_percent": round(holes, 1),
            "roundtrip_rmse_by_interpolator": errs,
            "point_pair_fit_max_error_px": {"affine": fit_err[0], "homography": fit_err[1]}}
    with open(os.path.join(OUT, "theory_numbers.json"), "w") as f:
        json.dump(info, f, indent=2)
    print(json.dumps(info, indent=2))
