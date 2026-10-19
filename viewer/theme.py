"""The marbling-bath world: palette, pigments, type and perceptual colour maps.

Colour maps are interpolated in OKLab between pigment stops, so lightness changes monotonically along each
map and equal steps look equal; error and data views never use the decorative pigments directly.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

FONT_DIR = Path(__file__).resolve().parent / "fonts"

# ground and materials
SIZE = "#c9ccbd"          # the size bath: the window ground
SIZE_WET = "#b7bdab"      # inside the tray, where the pigment floats
SIZE_DEEP = "#a2a993"     # tray edge, the deeper water
RIM = "#d5d7ca"           # the tray's lip and the rims of the window
PAPER = "#e6e3d6"         # a pulled sheet
STONE = "#8d927f"         # obstacles and outside-the-fluid regions
INK = "#141414"
INK_SOFT = "#3d4438"      # secondary text, tinted from the size (6:1 on SIZE)
HAIR = "#4b5243"          # hairline rules and live-derived numbers (5.0:1 on SIZE, 5.6:1 on RIM)
MADDER_TEXT = "#8a2232"   # madder deepened for text and the failed stamp (5.5:1 on SIZE)

SMALL = 13                # the smallest text anywhere in the window, in px (a projector across a room)
MOTION_MS = 200           # every state change eases out over this long (150-250 ms in the contract)

# the marbler's pigments, one per scene (spot colour) in benchmark order
PIGMENTS: Dict[str, str] = {
    "indigo": "#24366b",
    "malachite": "#2f6b55",
    "umber": "#6b4a2b",
    "ochre": "#a97b22",
    "madder": "#9e2b3a",
    "cobalt violet": "#5e3f73",
    "verdigris": "#2f7d74",
    "vermilion": "#b4462c",
}
SCENE_PIGMENT = {
    "tgv2d": "indigo",
    "cavity_ok": "malachite",
    "cavity_hard": "umber",
    "inverse": "ochre",
    "cylinder_fwd": "madder",
    "tgv3d": "cobalt violet",
    "deeponet": "verdigris",
    "stenosis": "vermilion",
}
# the order drops cycle through when the presenter adds pigment
DROP_ORDER = ["madder", "indigo", "ochre", "malachite", "cobalt violet", "umber", "verdigris"]


def hex_rgb(h: str) -> np.ndarray:
    h = h.lstrip("#")
    return np.array([int(h[i : i + 2], 16) for i in (0, 2, 4)], dtype=np.float64)


# ---------------------------------------------------------------------------- OKLab
_M1 = np.array([[0.4122214708, 0.5363325363, 0.0514459929], [0.2119034982, 0.6806995451, 0.1073969566], [0.0883024619, 0.2817188376, 0.6299787005]])
_M2 = np.array([[0.2104542553, 0.7936177850, -0.0040720468], [1.9779984951, -2.4285922050, 0.4505937099], [0.0259040371, 0.7827717662, -0.8086757660]])
_M2I = np.array([[1.0, 0.3963377774, 0.2158037573], [1.0, -0.1055613458, -0.0638541728], [1.0, -0.0894841775, -1.2914855480]])
_M1I = np.array([[4.0767416621, -3.3077115913, 0.2309699292], [-1.2684380046, 2.6097574011, -0.3413193965], [-0.0041960863, -0.7034186147, 1.7076147010]])


def _to_linear(c):
    c = c / 255.0
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _to_srgb(c):
    c = np.clip(c, 0.0, 1.0)
    return np.where(c <= 0.0031308, 12.92 * c, 1.055 * np.power(c, 1 / 2.4) - 0.055) * 255.0


def to_oklab(rgb):
    return np.cbrt(_to_linear(np.asarray(rgb, float)) @ _M1.T) @ _M2.T


def from_oklab(lab):
    return _to_srgb(((np.asarray(lab, float) @ _M2I.T) ** 3) @ _M1I.T)


def ramp(stops: Sequence[str], n: int = 256) -> np.ndarray:
    """(n, 3) uint8 colour map through the given stops, interpolated in OKLab."""
    labs = np.stack([to_oklab(hex_rgb(s)) for s in stops])
    x = np.linspace(0, len(stops) - 1, n)
    i = np.minimum(np.floor(x).astype(int), len(stops) - 2)
    f = (x - i)[:, None]
    return np.clip(np.round(from_oklab(labs[i] * (1 - f) + labs[i + 1] * f)), 0, 255).astype(np.uint8)


MAPS: Dict[str, np.ndarray] = {
    # magnitude: pale size -> verdigris -> indigo -> deep ink
    "sequential": ramp(["#ecebe2", "#93b2a3", "#2f7d74", "#24366b", "#11162a"]),
    # signed quantities: madder below zero, indigo above, pale size at zero
    "diverging": ramp(["#5c1420", "#9e2b3a", "#d6a49b", "#ecebe2", "#a3b3cc", "#24366b", "#101a3a"]),
    # error magnitude: pale -> ochre -> madder -> near-black lake
    "error": ramp(["#ecebe2", "#d9c08a", "#a97b22", "#9e2b3a", "#3b1016"]),
    # the bath's own sheen in the pigment view: speed as a faint deepening of the water
    "sheen": ramp([SIZE_WET, "#a7af98", "#949c84"]),
}

VIEWS = {
    "pigment": {"label": "pigment", "map": "sheen", "signed": False},
    "speed": {"label": "speed", "map": "sequential", "signed": False},
    "vorticity": {"label": "vorticity", "map": "diverging", "signed": True},
    "pressure": {"label": "pressure", "map": "diverging", "signed": True},
    "divergence": {"label": "divergence", "map": "diverging", "signed": True},
    "error": {"label": "error", "map": "error", "signed": False},
    "sensors": {"label": "sensors", "map": "sheen", "signed": False},
    "mri": {"label": "MRI scan", "map": "sequential", "signed": False},
}


# ---------------------------------------------------------------------------- type
_FAMILIES: Dict[str, str] = {}


def load_fonts() -> Dict[str, str]:
    """Register the bundled faces with Qt (and matplotlib). Returns {'display': family, 'text': family}."""
    from PySide6.QtGui import QFontDatabase

    fam = {"display": "Georgia", "text": "Segoe UI"}
    for key, fname in (("display", "BodoniModa[opsz,wght].ttf"), ("text", "LibreFranklin[wght].ttf")):
        path = FONT_DIR / fname
        if path.exists():
            fid = QFontDatabase.addApplicationFont(str(path))
            names = QFontDatabase.applicationFontFamilies(fid) if fid >= 0 else []
            if names:
                fam[key] = names[0]
    italic = FONT_DIR / "BodoniModa-Italic[opsz,wght].ttf"
    if italic.exists():
        QFontDatabase.addApplicationFont(str(italic))
    try:  # matplotlib cannot choose weights inside a variable font: give it static instances
        from matplotlib import font_manager

        for f in static_instances():
            font_manager.fontManager.addfont(str(f))
    except Exception:
        pass
    _FAMILIES.update(fam)
    return fam


def static_instances() -> List[Path]:
    """Regular and semibold static cuts of Libre Franklin, made once with fontTools' instancer."""
    out_dir = Path(__file__).resolve().parents[1] / "runs" / "_viewer_cache" / "fonts"
    out_dir.mkdir(parents=True, exist_ok=True)
    src = FONT_DIR / "LibreFranklin[wght].ttf"
    paths = []
    for wght in (400, 600):
        dst = out_dir / f"LibreFranklin-{wght}.ttf"
        if not dst.exists() and src.exists():
            from fontTools.ttLib import TTFont
            from fontTools.varLib.instancer import instantiateVariableFont

            inst = instantiateVariableFont(TTFont(str(src)), {"wght": wght}, updateFontNames=True)
            inst["OS/2"].usWeightClass = wght
            inst.save(str(dst))
        if dst.exists():
            paths.append(dst)
    return paths


def font(kind: str = "text", px: float = 13, weight: int = 400, tabular: bool = False, italic: bool = False, opsz: float | None = None):
    from PySide6.QtGui import QFont

    f = QFont(_FAMILIES.get(kind, "Segoe UI"))
    f.setPixelSize(int(round(px)))
    f.setWeight(QFont.Weight(weight))
    f.setItalic(italic)
    f.setHintingPreference(QFont.PreferNoHinting)
    try:
        f.setVariableAxis(QFont.Tag("wght"), float(weight))
        if kind == "display":
            f.setVariableAxis(QFont.Tag("opsz"), float(opsz or min(max(px, 6), 96)))
    except Exception:
        pass
    if tabular:
        try:
            f.setFeature(QFont.Tag("tnum"), 1)
            f.setFeature(QFont.Tag("lnum"), 1)
        except Exception:
            pass
    return f


def sci(v: float, digits: int = 1) -> str:
    """One notation for every small number: 4.2e-5 (no zero-padded exponent), both measured and live."""
    m, e = f"{v:.{digits}e}".split("e")
    return f"{m}e{int(e)}"


def qcolor(h: str, alpha: float = 1.0):
    from PySide6.QtGui import QColor

    c = QColor(h)
    c.setAlphaF(alpha)
    return c


def mpl_style() -> Dict:
    return {
        "font.family": _FAMILIES.get("text", "sans-serif"),
        "font.size": 9.5,            # points at 100 dpi: 13 px
        "mathtext.fontset": "custom",  # log-axis exponents in the same face as every other figure
        "mathtext.rm": _FAMILIES.get("text", "sans-serif"),
        "mathtext.it": _FAMILIES.get("text", "sans-serif"),
        "mathtext.bf": _FAMILIES.get("text", "sans-serif"),
        "mathtext.default": "regular",
        "axes.facecolor": PAPER,
        "figure.facecolor": SIZE,
        "axes.edgecolor": HAIR,
        "axes.labelcolor": INK_SOFT,
        "xtick.color": INK_SOFT,
        "ytick.color": INK_SOFT,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "legend.frameon": False,
        "axes.titlesize": 10,
    }


QSS = f"""
QMainWindow, QWidget#ground {{ background: {SIZE}; }}
QToolTip {{ background: {PAPER}; color: {INK}; border: 1px solid {HAIR}; padding: 4px 6px; }}
QPushButton#action {{
    background: {INK}; color: {PAPER}; border: none; border-radius: 4px; padding: 8px 14px;
}}
QPushButton#action:hover {{ background: #2a2f27; }}
QPushButton#action:focus {{ outline: none; border: 2px solid {PIGMENTS['ochre']}; }}
"""
