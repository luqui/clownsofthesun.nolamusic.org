"""Render raw band photos as matched, warm-toned black-and-white portraits.

Usage: python3 tools/portraits.py
"""

from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "photos" / "raw"
OUT = ROOT / "photos"
SIZE = 1000

# (filename, square crop box as left, top, side) in raw pixel coordinates
PORTRAITS = {
    "colman-grinell": ("colman-grinell.jpg", (250, 0, 619)),
    "luke-palmer": ("luke-palmer.jpg", (1150, 450, 1850)),
    "nick-benoit": ("nick-benoit.png", (384, 893, 773)),
}

SHADOW = np.array((14, 13, 12), dtype=float)
HIGHLIGHT = np.array((238, 232, 220), dtype=float)


def tone(lum):
    lo, hi = np.percentile(lum, (0.5, 99.5))
    stretched = np.clip((lum - lo) / (hi - lo), 0, 1)
    ranks = np.argsort(np.argsort(lum, axis=None)).reshape(lum.shape)
    equalized = ranks / (lum.size - 1)
    t = 0.75 * stretched + 0.25 * equalized
    return t * t * (3 - 2 * t) * 0.75 + t * 0.25


def vignette(shape):
    ys, xs = np.mgrid[-1 : 1 : shape[0] * 1j, -1 : 1 : shape[1] * 1j]
    return 1 - 0.22 * np.clip(np.hypot(xs, ys) - 0.5, 0, None) ** 1.5


def grain(shape, seed=0):
    noise = np.random.default_rng(seed).standard_normal(shape)
    return ndimage.gaussian_filter(noise, 0.6)


def portrait(src, box):
    left, top, side = box
    img = Image.open(src).convert("RGB").crop((left, top, left + side, top + side))
    rgb = np.asarray(img.resize((SIZE, SIZE), Image.LANCZOS), dtype=float) / 255
    lum = rgb @ np.array([0.45, 0.42, 0.13])

    t = np.clip(tone(lum) * vignette(lum.shape) + 0.035 * grain(lum.shape), 0, 1)
    out = SHADOW + (HIGHLIGHT - SHADOW) * t[..., None]
    return Image.fromarray(out.astype(np.uint8))


def main():
    for name, (filename, box) in PORTRAITS.items():
        portrait(RAW / filename, box).save(OUT / f"{name}.jpg", quality=88, progressive=True)
        print(f"wrote photos/{name}.jpg")


if __name__ == "__main__":
    main()
