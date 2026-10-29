"""Paint the hero background: a swirling Starry Night sky with a blazing sun.

Usage: python3 tools/sky.py
"""

import numpy as np
from PIL import Image

from paint import ROOT, brushstrokes

WIDTH, HEIGHT = 1800, 900

NIGHT = np.array((12, 20, 52))
ULTRAMARINE = np.array((27, 52, 120))
COBALT = np.array((52, 104, 170))
PALE = np.array((170, 200, 225))
CHROME = np.array((246, 204, 60))
CREAM = np.array((252, 240, 180))
CYPRESS = np.array((18, 38, 28))

# (x, y, radius) of bodies with ringed halos; the first is the sun.
LIGHTS = [(1480, 190, 95), (230, 150, 34), (560, 90, 26), (930, 120, 30), (1180, 330, 22), (420, 380, 18)]
VORTICES = [(820, 430, 170, 1.0), (1150, 560, 110, -1.0)]


def lerp(a, b, t):
    return a + (b - a) * t[..., None]


def cypress_mask(xs, ys):
    top = 90
    height = np.clip((ys - top) / (HEIGHT - top), 0, 1)
    flicker = 1 + 0.18 * np.sin(ys / 31 + 3 * np.sin(xs / 17))
    half_width = 105 * height**0.6 * flicker
    return np.abs(xs - 160 - 25 * np.sin(ys / 90)) < half_width


def main():
    ys, xs = np.mgrid[0:HEIGHT, 0:WIDTH].astype(float)

    band = 470 + 70 * np.sin(xs / 260)
    fx = np.ones_like(xs)
    fy = 0.35 * np.cos(xs / 260)
    light = 0.55 * ys / HEIGHT + 0.45 * np.exp(-(((ys - band) / 70) ** 2))
    glow = np.zeros_like(xs)

    for cx, cy, r, spin in VORTICES:
        dx, dy = xs - cx, ys - cy
        d = np.hypot(dx, dy) + 1e-6
        w = 3 * np.exp(-((d / r) ** 2))
        fx += spin * w * -dy / d
        fy += spin * w * dx / d
        light += 0.5 * np.exp(-((d / r) ** 2)) * (0.6 + 0.4 * np.cos(d / 14))

    for cx, cy, r in LIGHTS:
        dx, dy = xs - cx, ys - cy
        d = np.hypot(dx, dy) + 1e-6
        halo = np.exp(-((d / (2.6 * r)) ** 2))
        fx += 4 * halo * -dy / d
        fy += 4 * halo * dx / d
        glow = np.maximum(glow, np.where(d < r, 1.0, halo * (0.65 + 0.35 * np.cos((d - r) / (r / 3.5)))))

    tree = cypress_mask(xs, ys)
    fx[tree], fy[tree] = 0.25 * np.sin(ys[tree] / 30), -1.0

    norm = np.hypot(fx, fy)
    strokes = brushstrokes(fx / norm, fy / norm, length=12, grain=1.4, seed=3)

    light = np.clip(light + 0.08 * strokes, 0, 1)
    sky = np.where(
        (light < 0.5)[..., None],
        lerp(NIGHT, ULTRAMARINE, light / 0.5),
        lerp(COBALT, PALE, (light - 0.5) / 0.5),
    )
    sky = lerp(sky, lerp(CHROME, CREAM, np.clip(glow - 0.6, 0, 1) / 0.4), np.clip(glow + 0.05 * strokes, 0, 1))
    sky[tree] = CYPRESS * (1 + 0.6 * np.clip(strokes[tree], -1, 2))[..., None]
    sky *= 1 + 0.12 * strokes[..., None]

    Image.fromarray(np.clip(sky, 0, 255).astype(np.uint8)).save(ROOT / "art" / "sky.jpg", quality=85)
    print("wrote art/sky.jpg")


if __name__ == "__main__":
    main()
