"""Render raw band photos as consistent Van Gogh-style portraits.

Usage: python3 tools/paint.py
"""

from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "photos" / "raw"
OUT = ROOT / "photos"
SIZE = 900

# (filename, square crop box as left, top, side) in raw pixel coordinates
PORTRAITS = {
    "colman-grinell": ("colman-grinell.jpg", (250, 0, 619)),
    "luke-palmer": ("luke-palmer.jpg", (1150, 450, 1850)),
    "nick-benoit": ("nick-benoit.png", (384, 893, 773)),
}

# Starry Night gradient: night ultramarine through cobalt and teal to chrome yellow.
PALETTE = np.array(
    [
        (0.00, (12, 20, 52)),
        (0.25, (27, 52, 120)),
        (0.45, (46, 98, 158)),
        (0.60, (86, 140, 150)),
        (0.75, (196, 170, 78)),
        (0.90, (242, 204, 64)),
        (1.00, (252, 238, 170)),
    ],
    dtype=object,
)


def kuwahara(img, radius):
    size = radius + 1
    means, variances = [], []
    lum = img.mean(axis=2)
    for dy in (-radius // 2, radius // 2):
        for dx in (-radius // 2, radius // 2):
            shift = lambda a: ndimage.shift(a, (dy, dx) + (0,) * (a.ndim - 2), mode="nearest", order=0)
            mean = shift(ndimage.uniform_filter(img, (size, size, 1)))
            sq = shift(ndimage.uniform_filter(lum**2, size))
            means.append(mean)
            variances.append(sq - shift(ndimage.uniform_filter(lum, size)) ** 2)
    best = np.argmin(np.stack(variances), axis=0)
    return np.take_along_axis(np.stack(means), best[None, ..., None], axis=0)[0]


def structure_tangent(lum):
    gy, gx = np.gradient(ndimage.gaussian_filter(lum, 2))
    jxx = ndimage.gaussian_filter(gx * gx, 6)
    jxy = ndimage.gaussian_filter(gx * gy, 6)
    jyy = ndimage.gaussian_filter(gy * gy, 6)
    angle = 0.5 * np.arctan2(2 * jxy, jxx - jyy) + np.pi / 2
    return np.cos(angle), np.sin(angle)


def brushstrokes(tx, ty, length=10, grain=0.8, seed=0):
    """Line integral convolution of noise along the unit field (tx, ty), normalized."""
    shape = tx.shape
    noise = ndimage.gaussian_filter(np.random.default_rng(seed).random(shape), grain)
    ys, xs = np.mgrid[0 : shape[0], 0 : shape[1]].astype(float)
    total = noise.copy()
    for direction in (1, -1):
        px, py = xs.copy(), ys.copy()
        for _ in range(length):
            ix = np.clip(px, 0, shape[1] - 1).astype(int)
            iy = np.clip(py, 0, shape[0] - 1).astype(int)
            px += direction * tx[iy, ix]
            py += direction * ty[iy, ix]
            total += ndimage.map_coordinates(noise, (py, px), order=1, mode="reflect")
    total /= 2 * length + 1
    return (total - total.mean()) / total.std()


def stretch(lum):
    lo, hi = np.percentile(lum, (1, 99))
    return np.clip((lum - lo) / (hi - lo), 0, 1)


def equalize(lum):
    ranks = np.argsort(np.argsort(lum, axis=None))
    return (ranks / (lum.size - 1)).reshape(lum.shape)


def gradient_map(t):
    stops = np.array([s for s, _ in PALETTE], dtype=float)
    colors = np.array([c for _, c in PALETTE], dtype=float)
    return np.stack([np.interp(t, stops, colors[:, i]) for i in range(3)], axis=-1)


def paint(src, box):
    left, top, side = box
    img = Image.open(src).convert("RGB").crop((left, top, left + side, top + side))
    img = np.asarray(img.resize((SIZE, SIZE), Image.LANCZOS), dtype=float) / 255

    smooth = kuwahara(img, 4)
    lum = smooth @ np.array([0.299, 0.587, 0.114])
    lum = 0.5 * equalize(lum) + 0.5 * stretch(lum)

    strokes = brushstrokes(*structure_tangent(lum))
    lum = np.clip(lum + 0.04 * strokes, 0, 1)
    out = gradient_map(lum) * (1 + 0.06 * strokes[..., None])
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


def main():
    for name, (filename, box) in PORTRAITS.items():
        paint(RAW / filename, box).save(OUT / f"{name}.jpg", quality=88)
        print(f"wrote photos/{name}.jpg")


if __name__ == "__main__":
    main()
