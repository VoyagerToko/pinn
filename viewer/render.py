"""Turning a served field into pixels: scalar per view, fixed colour scale per scene, NaN as stone."""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

from viewer import theme

STONE = theme.hex_rgb(theme.STONE).astype(np.uint8)


def scalar(fields: Dict[str, np.ndarray], view: str, ref: bool = False) -> Optional[np.ndarray]:
    """The quantity a view shows, from the prediction or (ref=True) the reference. None when not served."""
    sfx = "_ref" if ref else ""
    if view == "mri":  # the scan's voxel velocities; its reference is the CFD speed it averages
        u, v = (fields.get("u_ref"), fields.get("v_ref")) if ref else (fields.get("u_mri"), fields.get("v_mri"))
        return None if u is None or v is None else np.hypot(u, v)
    if view in ("pigment", "speed", "sensors"):
        if "speed" in fields and not ref:
            return fields["speed"]
        u, v = fields.get("u" + sfx), fields.get("v" + sfx)
        return None if u is None or v is None else np.hypot(u, v)
    if view == "vorticity":
        return fields.get("vort" + sfx)
    if view == "pressure":
        return fields.get("p" + sfx)
    if view == "divergence":
        return None if ref else fields.get("div")
    if view == "error":
        if ref or "u_ref" not in fields:
            return None
        return np.hypot(fields["u"] - fields["u_ref"], fields["v"] - fields["v_ref"])
    return None


def colour_map(view: str) -> np.ndarray:
    key = "sheen" if view in ("pigment", "sensors") else theme.VIEWS.get(view, {}).get("map", "sequential")
    return theme.MAPS[key]


def signed(view: str) -> bool:
    return theme.VIEWS.get(view, {}).get("signed", False)


def scale_for(values, view: str) -> Tuple[float, float]:
    vals = np.concatenate([np.asarray(v, np.float32).ravel() for v in values if v is not None])
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return 0.0, 1.0
    if signed(view):
        m = float(np.percentile(np.abs(vals), 99.5)) or 1.0
        return -m, m
    return 0.0, float(np.percentile(vals, 99.5)) or 1.0


def to_rgb(a: np.ndarray, view: str, lo: float, hi: float) -> np.ndarray:
    """(nx, ny) 'ij' field -> (ny, nx, 3) uint8 image with y up, NaN painted as stone."""
    lut = colour_map(view)
    a = np.asarray(a, np.float32)
    nan = ~np.isfinite(a)
    x = np.clip((np.nan_to_num(a) - lo) / max(hi - lo, 1e-30), 0.0, 1.0)
    img = lut[(x * (len(lut) - 1)).astype(np.int32)]
    img[nan] = STONE
    return np.ascontiguousarray(img.transpose(1, 0, 2)[::-1])


def qimage(rgb: np.ndarray):
    from PySide6.QtGui import QImage

    h, w = rgb.shape[:2]
    if rgb.shape[2] == 3:
        return QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888).copy()
    return QImage(rgb.data, w, h, 4 * w, QImage.Format_ARGB32_Premultiplied).copy()


def rel_l2(fields: Dict[str, np.ndarray]) -> Optional[float]:
    if "u_ref" not in fields or "u" not in fields:
        return None
    pu, pv, ru, rv = (fields[k] for k in ("u", "v", "u_ref", "v_ref"))
    m = np.isfinite(pu) & np.isfinite(ru)
    num = np.sqrt(np.sum((pu[m] - ru[m]) ** 2 + (pv[m] - rv[m]) ** 2))
    den = np.sqrt(np.sum(ru[m] ** 2 + rv[m] ** 2))
    return float(num / den) if den > 0 else None
