"""
Batched GPU data augmentation built on the same maths as geometry.py.

Geometric: every image gets its own random 3x3 homogeneous matrix
    H = C · P · T · R · Sh · S · F · C^-1     (C = move origin to image centre)
    F  = horizontal flip      S  = scale        Sh = shear
    R  = rotation             T  = translation  P  = projective (homography) term
and is resampled by INVERSE warping: for each output pixel, (x, y) = H^-1 (x', y'),
then the input is interpolated there with nearest / bilinear / bicubic (grid_sample).

Photometric: brightness, contrast, saturation, gaussian noise, random grayscale.
"""
import math
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------- geometric

def _eye(B, dev):
    return torch.eye(3, device=dev).repeat(B, 1, 1)


def random_homographies(B, size, dev, rot=15.0, scale=(0.85, 1.15), translate=0.125,
                        shear=10.0, persp=0.0008, flip=True):
    """Return B random 3x3 matrices in PIXEL coordinates of a size x size image."""
    U = lambda lo, hi: torch.empty(B, device=dev).uniform_(lo, hi)
    c = (size - 1) / 2.0

    Fm = _eye(B, dev)
    if flip:
        Fm[:, 0, 0] = torch.where(torch.rand(B, device=dev) < 0.5, -1.0, 1.0)
    S = _eye(B, dev)
    s = U(*scale)
    S[:, 0, 0], S[:, 1, 1] = s, s
    Sh = _eye(B, dev)
    Sh[:, 0, 1] = torch.tan(torch.deg2rad(U(-shear, shear)))
    R = _eye(B, dev)
    t = torch.deg2rad(U(-rot, rot))
    R[:, 0, 0], R[:, 0, 1], R[:, 1, 0], R[:, 1, 1] = t.cos(), -t.sin(), t.sin(), t.cos()
    T = _eye(B, dev)
    T[:, 0, 2], T[:, 1, 2] = U(-translate, translate) * size, U(-translate, translate) * size
    P = _eye(B, dev)
    P[:, 2, 0], P[:, 2, 1] = U(-persp, persp), U(-persp, persp)
    C, Ci = _eye(B, dev), _eye(B, dev)
    C[:, 0, 2] = C[:, 1, 2] = c
    Ci[:, 0, 2] = Ci[:, 1, 2] = -c
    return C @ P @ T @ R @ Sh @ S @ Fm @ Ci


def inverse_warp(x, H, mode="bilinear"):
    """x: (B,C,h,w). H: (B,3,3) source->destination in pixel coords.
    For every destination pixel apply H^-1, then interpolate the source there."""
    B, _, h, w = x.shape
    dev = x.device
    ys, xs = torch.meshgrid(torch.arange(h, device=dev, dtype=torch.float32),
                            torch.arange(w, device=dev, dtype=torch.float32), indexing="ij")
    dst = torch.stack([xs.flatten(), ys.flatten(), torch.ones(h * w, device=dev)])  # 3 x N
    src = torch.linalg.inv(H) @ dst                                                 # B x 3 x N
    sx, sy = src[:, 0] / src[:, 2], src[:, 1] / src[:, 2]
    # pixel coords -> [-1, 1] (align_corners=True convention)
    grid = torch.stack([2 * sx / (w - 1) - 1, 2 * sy / (h - 1) - 1], -1).view(B, h, w, 2)
    out = F.grid_sample(x, grid, mode=mode, padding_mode="zeros", align_corners=True)
    return out.clamp(0, 1) if mode == "bicubic" else out


def fixed_homography(B, size, dev, rot=0.0, scale=1.0, tx=0.0, ty=0.0, shear=0.0,
                     persp=(0.0, 0.0), flip=False):
    """Same composition as random_homographies but with fixed parameters (for test-time shifts)."""
    c = (size - 1) / 2.0
    t =math.radians(rot)
    R = torch.tensor([[math.cos(t), -math.sin(t), 0], [math.sin(t), math.cos(t), 0], [0, 0, 1]], device=dev)
    S = torch.diag(torch.tensor([scale * (-1 if flip else 1), scale, 1.0], device=dev))
    Sh = torch.eye(3, device=dev); Sh[0, 1] = math.tan(math.radians(shear))
    T = torch.eye(3, device=dev); T[0, 2], T[1, 2] = tx, ty
    P = torch.eye(3, device=dev); P[2, 0], P[2, 1] = persp
    C = torch.eye(3, device=dev); C[0, 2] = C[1, 2] = c
    Ci = torch.eye(3, device=dev); Ci[0, 2] = Ci[1, 2] = -c
    M = C @ P @ T @ R @ Sh @ S @ Ci
    return M.expand(B, 3, 3).contiguous()


# ---------------------------------------------------------------- photometric

def _gray(x):
    return (0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3])


def adjust(x, brightness=1.0, contrast=1.0, saturation=1.0):
    """Factors may be floats or (B,) tensors."""
    def bc(v):
        return v.view(-1, 1, 1, 1) if torch.is_tensor(v) else v
    x = x * bc(brightness)
    m = _gray(x).mean(dim=(1, 2, 3), keepdim=True)
    x = (x - m) * bc(contrast) + m
    g = _gray(x)
    x = (x - g) * bc(saturation) + g
    return x.clamp(0, 1)


def random_photometric(x, b=0.3, c=0.3, s=0.3, noise=0.04, p_gray=0.1):
    B, dev = x.shape[0], x.device
    U = lambda lo, hi: torch.empty(B, device=dev).uniform_(lo, hi)
    x = adjust(x, U(1 - b, 1 + b), U(1 - c, 1 + c), U(1 - s, 1 + s))
    sigma = U(0, noise).view(-1, 1, 1, 1)
    x = x + sigma * torch.randn_like(x)
    gray = (torch.rand(B, device=dev) < p_gray).view(-1, 1, 1, 1)
    x = torch.where(gray, _gray(x).expand_as(x), x)
    return x.clamp(0, 1)


# ---------------------------------------------------------------- pipeline

class Augmenter:
    def __init__(self, geometric=False, photometric=False, interp="bilinear"):
        self.geometric, self.photometric, self.interp = geometric, photometric, interp

    def __call__(self, x):
        if self.geometric:
            H = random_homographies(x.shape[0], x.shape[-1], x.device)
            x = inverse_warp(x, H, self.interp)
        if self.photometric:
            x = random_photometric(x)
        return x


def gaussian_blur(x, sigma=1.0):
    """Used only at TEST time: a corruption the model never saw during training."""
    k = int(2 * math.ceil(2 * sigma) + 1)
    r = torch.arange(k, device=x.device, dtype=torch.float32) - k // 2
    g = torch.exp(-r ** 2 / (2 * sigma ** 2)); g = g / g.sum()
    C = x.shape[1]
    x = F.conv2d(F.pad(x, (k // 2,) * 4, mode="reflect"), g.view(1, 1, 1, k).repeat(C, 1, 1, 1), groups=C)
    x = F.conv2d(x, g.view(1, 1, k, 1).repeat(C, 1, 1, 1), groups=C)
    return x
