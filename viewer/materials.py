"""Materials of the bath, made procedurally at run time (no raster files ship with the app).

paper_sheet   a pulled sheet: deckled alpha edge with fibre, faint grain, and its blurred shadow
soft_shadow   a blurred drop shadow for any rounded shape (the tray, the stamp)
ink_texture   uneven stamp-ink density, used as an alpha mask over the stamp's impression

Everything is cached per pixel size, so resizing the window re-makes each material once.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Tuple

import numpy as np
from scipy.ndimage import gaussian_filter

from viewer import theme


def _qimage_rgba(a: np.ndarray):
    """(H, W, 4) RGBA uint8 -> premultiplied ARGB32 QImage."""
    from PySide6.QtGui import QImage

    rgba = a.astype(np.float32)
    al = rgba[..., 3:4] / 255.0
    out = np.empty_like(a)
    out[..., 0] = np.clip(rgba[..., 2] * al[..., 0], 0, 255)
    out[..., 1] = np.clip(rgba[..., 1] * al[..., 0], 0, 255)
    out[..., 2] = np.clip(rgba[..., 0] * al[..., 0], 0, 255)
    out[..., 3] = a[..., 3]
    out = np.ascontiguousarray(out)
    h, w = out.shape[:2]
    return QImage(out.data, w, h, 4 * w, QImage.Format_ARGB32_Premultiplied).copy()


def _edge_noise(n: int, rng: np.random.Generator, amp: float) -> np.ndarray:
    """1-D fibrous wander along an edge: a few octaves of smoothed noise."""
    out = np.zeros(n, np.float32)
    for sigma, a in ((40, 1.0), (10, 0.55), (2.5, 0.35)):
        out += a * gaussian_filter(rng.standard_normal(n).astype(np.float32), sigma, mode="wrap")
    out /= np.abs(out).max() + 1e-6
    return amp * out


@lru_cache(maxsize=16)
def paper_sheet(w: int, h: int, seed: int = 11, margin: int = 14):
    """(sheet QImage, shadow QImage) of size (w, h) device pixels; the paper sits inside `margin`."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    left = _edge_noise(h, rng, 2.6)[:, None]
    right = _edge_noise(h, rng, 2.6)[:, None]
    top = _edge_noise(w, rng, 2.6)[None, :]
    bottom = _edge_noise(w, rng, 2.6)[None, :]
    d = np.minimum.reduce([xx - margin - left, (w - margin) - xx - right, yy - margin - top, (h - margin) - yy - bottom])
    fibre = gaussian_filter(rng.standard_normal((h, w)).astype(np.float32), 0.7)
    alpha = np.clip((d + 0.9 * fibre) / 2.2, 0.0, 1.0)
    alpha *= np.clip(1.0 - np.exp(-np.maximum(d, 0) / 1.2), 0.0, 1.0) * 0.25 + 0.75  # translucent torn edge
    grain = gaussian_filter(rng.standard_normal((h, w)).astype(np.float32), 1.2)
    grain = grain / (np.abs(grain).max() + 1e-6)
    base = theme.hex_rgb(theme.PAPER).astype(np.float32)
    rgb = base[None, None, :] * (1.0 + 0.028 * grain[..., None])
    sheet = np.dstack([np.clip(rgb, 0, 255), alpha * 255]).astype(np.uint8)
    sh = gaussian_filter(np.roll(np.roll((d > 0).astype(np.float32), 4, axis=0), 2, axis=1), 6.0) * 0.20
    ink = theme.hex_rgb(theme.INK)
    shadow = np.dstack([np.broadcast_to(ink, (h, w, 3)), np.clip(sh * 255, 0, 255)]).astype(np.uint8)
    return _qimage_rgba(sheet), _qimage_rgba(shadow)


@lru_cache(maxsize=16)
def soft_shadow(w: int, h: int, radius: float, inset: int, blur: float = 7.0, dy: int = 5, opacity: float = 0.22):
    """Blurred drop shadow of a rounded rectangle (inset from the image edge), offset downwards by dy."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    x0, y0, x1, y1 = inset, inset, w - inset, h - inset
    cx = np.clip(xx, x0 + radius, x1 - radius)
    cy = np.clip(yy, y0 + radius, y1 - radius)
    inside = ((xx - cx) ** 2 + (yy - cy) ** 2 <= radius**2) & (xx >= x0) & (xx <= x1) & (yy >= y0) & (yy <= y1)
    m = gaussian_filter(np.roll(inside.astype(np.float32), dy, axis=0), blur) * opacity
    ink = theme.hex_rgb(theme.INK)
    return _qimage_rgba(np.dstack([np.broadcast_to(ink, (h, w, 3)), np.clip(m * 255, 0, 255)]).astype(np.uint8))


@lru_cache(maxsize=8)
def ink_texture(w: int, h: int, seed: int = 5):
    """Alpha mask of uneven ink: mostly full, softly mottled, with a few dry specks."""
    rng = np.random.default_rng(seed)
    blot = gaussian_filter(rng.standard_normal((h, w)).astype(np.float32), 6.0)
    blot = (blot - blot.min()) / (np.ptp(blot) + 1e-6)
    speck = gaussian_filter(rng.standard_normal((h, w)).astype(np.float32), 0.8) > 2.1
    a = 0.80 + 0.20 * blot
    a[speck] *= 0.35
    img = np.dstack([np.zeros((h, w, 3), np.float32), np.clip(a, 0, 1) * 255]).astype(np.uint8)
    return _qimage_rgba(img)


def tint(hex_colour: str, toward: str, f: float) -> Tuple[float, float, float]:
    a, b = theme.hex_rgb(hex_colour), theme.hex_rgb(toward)
    return tuple((a * (1 - f) + b * f).tolist())
