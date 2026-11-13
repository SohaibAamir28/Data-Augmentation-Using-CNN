"""
From-scratch geometric transformations, warping and interpolation (NumPy only).

Follows Lecture 7 (Geometric Transformations):
  translation -> rotation -> Euclidean (rigid) -> similarity -> affine -> homography
  forward warping vs. inverse warping
  nearest-neighbour, bilinear and bicubic interpolation

All matrices are 3x3 in homogeneous coordinates and act on column vectors [x, y, 1]^T,
with x = column index and y = row index.
"""
import numpy as np


# ---------------------------------------------------------------------------
# 1. Transformation matrices (homogeneous coordinates)
# ---------------------------------------------------------------------------

def translation(tx, ty):
    """2 DoF."""
    return np.array([[1, 0, tx],
                     [0, 1, ty],
                     [0, 0, 1]], dtype=np.float64)


def rotation(theta_deg):
    """1 DoF (rotation about the origin)."""
    t = np.deg2rad(theta_deg)
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, -s, 0],
                     [s,  c, 0],
                     [0,  0, 1]], dtype=np.float64)


def euclidean(theta_deg, tx, ty):
    """Rigid body: rotation + translation, 3 DoF."""
    t = np.deg2rad(theta_deg)
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, -s, tx],
                     [s,  c, ty],
                     [0,  0, 1]], dtype=np.float64)


def similarity(scale, theta_deg, tx, ty):
    """Rotation + translation + uniform scale, 4 DoF."""
    t = np.deg2rad(theta_deg)
    a = scale
    return np.array([[a * np.cos(t), -a * np.sin(t), tx],
                     [a * np.sin(t),  a * np.cos(t), ty],
                     [0, 0, 1]], dtype=np.float64)


def affine(a, b, c, d, e, f):
    """6 DoF: translation + rotation + (non-uniform) scale + skew."""
    return np.array([[a, b, c],
                     [d, e, f],
                     [0, 0, 1]], dtype=np.float64)


def homography(a, b, c, d, e, f, g, h):
    """8 DoF projective transform (last element fixed to 1 -> defined up to scale)."""
    return np.array([[a, b, c],
                     [d, e, f],
                     [g, h, 1]], dtype=np.float64)


def about_center(M, w, h):
    """Apply M about the image centre instead of the top-left origin."""
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    return translation(cx, cy) @ M @ translation(-cx, -cy)


# ---------------------------------------------------------------------------
# 2. Solving for transforms from point correspondences
# ---------------------------------------------------------------------------

def solve_affine(src, dst):
    """Least-squares affine from >= 3 point pairs (6 unknowns, 2 equations per pair)."""
    src, dst = np.asarray(src, float), np.asarray(dst, float)
    n = len(src)
    assert n >= 3, "need at least 3 point pairs for 6 unknowns"
    A = np.zeros((2 * n, 6))
    bvec = np.zeros(2 * n)
    for i, ((x, y), (xp, yp)) in enumerate(zip(src, dst)):
        A[2 * i] = [x, y, 1, 0, 0, 0]
        A[2 * i + 1] = [0, 0, 0, x, y, 1]
        bvec[2 * i], bvec[2 * i + 1] = xp, yp
    p, *_ = np.linalg.lstsq(A, bvec, rcond=None)
    return affine(*p)


def solve_homography(src, dst):
    """Direct Linear Transform from >= 4 point pairs (8 unknowns, 2 equations per pair)."""
    src, dst = np.asarray(src, float), np.asarray(dst, float)
    n = len(src)
    assert n >= 4, "need at least 4 point pairs for 8 unknowns"
    A = []
    for (x, y), (xp, yp) in zip(src, dst):
        A.append([-x, -y, -1, 0, 0, 0, x * xp, y * xp, xp])
        A.append([0, 0, 0, -x, -y, -1, x * yp, y * yp, yp])
    _, _, Vt = np.linalg.svd(np.asarray(A))
    H = Vt[-1].reshape(3, 3)
    return H / H[2, 2]


def apply_to_points(M, pts):
    pts = np.asarray(pts, float)
    ph = np.c_[pts, np.ones(len(pts))] @ M.T
    return ph[:, :2] / ph[:, 2:3]


# ---------------------------------------------------------------------------
# 3. Interpolation: sample image I at real-valued coordinates (xs, ys)
# ---------------------------------------------------------------------------

def _pixel(img, xi, yi, fill):
    """Fetch img[yi, xi] with out-of-bounds -> fill. Works for HxW or HxWxC."""
    h, w = img.shape[:2]
    inside = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
    xc, yc = np.clip(xi, 0, w - 1), np.clip(yi, 0, h - 1)
    out = img[yc, xc].astype(np.float64)
    mask = inside if img.ndim == 2 else inside[..., None]
    return np.where(mask, out, fill)


def interp_nearest(img, xs, ys, fill=0.0):
    """Copy the value of the closest source pixel (blocky)."""
    return _pixel(img, np.round(xs).astype(int), np.round(ys).astype(int), fill)


def interp_bilinear(img, xs, ys, fill=0.0):
    """4 neighbours: linear in x, then linear in y.
    f(x,y) = f00(1-dx)(1-dy) + f10 dx(1-dy) + f01(1-dx)dy + f11 dx dy
    """
    x0, y0 = np.floor(xs).astype(int), np.floor(ys).astype(int)
    dx, dy = xs - x0, ys - y0
    if img.ndim == 3:
        dx, dy = dx[..., None], dy[..., None]
    f00 = _pixel(img, x0, y0, fill)
    f10 = _pixel(img, x0 + 1, y0, fill)
    f01 = _pixel(img, x0, y0 + 1, fill)
    f11 = _pixel(img, x0 + 1, y0 + 1, fill)
    return (f00 * (1 - dx) * (1 - dy) + f10 * dx * (1 - dy)
            + f01 * (1 - dx) * dy + f11 * dx * dy)


def _cubic_weight(t, a=-0.5):
    """Keys cubic convolution kernel (a = -0.5, the standard 'bicubic')."""
    t = np.abs(t)
    w = np.zeros_like(t)
    m1 = t <= 1
    m2 = (t > 1) & (t < 2)
    w[m1] = (a + 2) * t[m1] ** 3 - (a + 3) * t[m1] ** 2 + 1
    w[m2] = a * t[m2] ** 3 - 5 * a * t[m2] ** 2 + 8 * a * t[m2] - 4 * a
    return w


def interp_bicubic(img, xs, ys, fill=0.0):
    """16 neighbours: a cubic polynomial fit along x for each of 4 rows, then along y."""
    x0, y0 = np.floor(xs).astype(int), np.floor(ys).astype(int)
    dx, dy = xs - x0, ys - y0
    out = 0.0
    for m in range(-1, 3):
        wy = _cubic_weight(dy - m)
        row = 0.0
        for n in range(-1, 3):
            wx = _cubic_weight(dx - n)
            wxx = wx[..., None] if img.ndim == 3 else wx
            row = row + wxx * _pixel(img, x0 + n, y0 + m, fill)
        wyy = wy[..., None] if img.ndim == 3 else wy
        out = out + wyy * row
    return out


INTERPOLATORS = {
    "nearest": interp_nearest,
    "bilinear": interp_bilinear,
    "bicubic": interp_bicubic,
}


# ---------------------------------------------------------------------------
# 4. Warping
# ---------------------------------------------------------------------------

def inverse_warp(img, M, out_shape=None, method="bilinear", fill=0.0):
    """For every DESTINATION pixel (x', y'), find where it came from:
    (x, y) = M^-1 (x', y'), then interpolate I at that real-valued location.
    Every output pixel gets exactly one value -> no holes.
    """
    h, w = out_shape if out_shape is not None else img.shape[:2]
    Minv = np.linalg.inv(M)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    pts = np.stack([xx.ravel(), yy.ravel(), np.ones(h * w)])
    src = Minv @ pts
    xs = (src[0] / src[2]).reshape(h, w)
    ys = (src[1] / src[2]).reshape(h, w)
    out = INTERPOLATORS[method](img, xs, ys, fill)
    return np.clip(out, 0, 255) if img.dtype == np.uint8 else out


def forward_warp(img, M, out_shape=None, fill=0.0):
    """For every SOURCE pixel (x, y), push its value to round(M (x, y)).
    Several sources can hit the same target and some targets get nothing -> holes.
    Returns (warped image, hole mask).
    """
    h, w = out_shape if out_shape is not None else img.shape[:2]
    H, W = img.shape[:2]
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    pts = np.stack([xx.ravel(), yy.ravel(), np.ones(H * W)])
    dst = M @ pts
    xd = np.round(dst[0] / dst[2]).astype(int)
    yd = np.round(dst[1] / dst[2]).astype(int)
    keep = (xd >= 0) & (xd < w) & (yd >= 0) & (yd < h)
    out = np.full((h, w) + img.shape[2:], fill, dtype=np.float64)
    hit = np.zeros((h, w), bool)
    vals = img.reshape(H * W, *img.shape[2:])
    out[yd[keep], xd[keep]] = vals[keep]
    hit[yd[keep], xd[keep]] = True
    return out, ~hit
