"""Controls in the bath's vocabulary. Every choice stays on screen; the active one is struck forward."""
from __future__ import annotations

import math
from typing import Dict, List, Optional

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QBrush, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QSizePolicy, QWidget

from viewer import theme
from viewer.scenes import SCENES


def blot(c: QPointF, r: float, seed: int) -> QPainterPath:
    """A pigment drop as it spreads on the size: a round blot with a few soft lobes."""
    rng = np.random.default_rng(seed)
    k = 7
    rad = r * (1 + 0.10 * rng.standard_normal(k))
    path = QPainterPath()
    ang = [2 * math.pi * i / k for i in range(k)]
    pts = [QPointF(c.x() + rad[i] * math.cos(a), c.y() + rad[i] * math.sin(a)) for i, a in enumerate(ang)]
    path.moveTo((pts[0] + pts[-1]) / 2)
    for i in range(k):
        nxt = pts[(i + 1) % k]
        path.quadTo(pts[i], (pts[i] + nxt) / 2)
    path.closeSubpath()
    return path


class SceneRail(QWidget):
    chosen = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(78)
        self.setMouseTracking(True)
        self.active = SCENES[0].sid
        self.ready: Dict[str, str] = {s.sid: "waiting" for s in SCENES}
        self.available = {s.sid for s in SCENES}
        self.hover = None

    def _cells(self) -> List[QRectF]:
        w = self.width() / len(SCENES)
        return [QRectF(i * w, 0, w, self.height()) for i in range(len(SCENES))]

    def mouseMoveEvent(self, e):
        h = next((s.sid for s, r in zip(SCENES, self._cells()) if r.contains(e.position())), None)
        if h != self.hover:
            self.hover = h
            self.setCursor(Qt.PointingHandCursor if h else Qt.ArrowCursor)
            self.update()

    def leaveEvent(self, e):
        self.hover = None
        self.update()

    def mousePressEvent(self, e):
        for s, r in zip(SCENES, self._cells()):
            if r.contains(e.position()) and s.sid in self.available:
                self.chosen.emit(s.sid)

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        for i, (s, r) in enumerate(zip(SCENES, self._cells())):
            on = s.sid == self.active
            avail = s.sid in self.available
            state = self.ready.get(s.sid, "waiting")
            if self.hover == s.sid and not on:
                p.setPen(Qt.NoPen)
                p.setBrush(theme.qcolor(theme.INK, 0.05))
                p.drawRoundedRect(r.adjusted(4, 4, -4, -8), 6, 6)
            c = QPointF(r.left() + 30, r.top() + 30)
            rad = 15.0 if on else 11.5
            path = blot(c, rad, seed=3 + i)
            if state == "ready" or on:
                p.setPen(Qt.NoPen)
                p.setBrush(theme.qcolor(s.pigment, 1.0 if on else 0.82))
            else:  # still compiling on the GPU: an unfilled drop
                p.setPen(QPen(theme.qcolor(s.pigment, 0.9), 1.6))
                p.setBrush(Qt.NoBrush)
            p.setOpacity(1.0 if avail else 0.35)
            p.drawPath(path)
            if on:
                p.setPen(QPen(theme.qcolor(theme.INK), 1.3))
                p.setBrush(Qt.NoBrush)
                p.drawPath(blot(c, rad + 4.5, seed=3 + i))
            x = r.left() + 54
            p.setFont(theme.font("text", 13.5, 650 if on else 500))
            p.setPen(theme.qcolor(theme.INK if on else theme.INK_SOFT))
            p.drawText(QRectF(x, r.top() + 12, r.width() - 60, 20), Qt.AlignLeft | Qt.AlignVCenter, s.short)
            p.setFont(theme.font("text", 11.5, 500))
            word = s.verdict if state in ("ready", "waiting") or on else ("compiling" if state == "warming" else state)
            p.setPen(theme.qcolor(theme.INK_SOFT if s.verdict != "failed" else theme.PIGMENTS["madder"]))
            p.drawText(QRectF(x, r.top() + 32, r.width() - 60, 18), Qt.AlignLeft | Qt.AlignVCenter, word)
            p.setOpacity(1.0)
            if on:
                p.setPen(Qt.NoPen)
                p.setBrush(theme.qcolor(s.pigment))
                p.drawRect(QRectF(r.left() + 10, r.bottom() - 6, r.width() - 20, 3))
        p.setPen(QPen(theme.qcolor(theme.INK, 0.25), 1))
        p.drawLine(QPointF(0, self.height() - 1), QPointF(self.width(), self.height() - 1))
        p.end()


class VerdictStamp(QWidget):
    """The verdict as an ink stamp: word, measured number in solid type, live-derived number in hairline."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(300, 112)
        self.word = ""
        self.number = ""
        self.label = ""
        self.live = ""
        self.pigment = theme.INK

    def set(self, word: str, number: str, label: str, pigment: str):
        self.word, self.number, self.label, self.pigment = word, number, label, pigment
        self.live = ""
        self.update()

    def set_live(self, text: str):
        if text != self.live:
            self.live = text
            self.update()

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        r = QRectF(self.rect()).adjusted(1, 1, -1, -1)
        failed = self.word == "failed"
        p.setPen(QPen(theme.qcolor(theme.PIGMENTS["madder"] if failed else theme.INK, 0.95), 1.6))
        p.setBrush(theme.qcolor(theme.PAPER, 0.55))
        p.drawRoundedRect(r, 5, 5)
        p.setFont(theme.font("display", 15, 600))
        p.setPen(theme.qcolor(theme.PIGMENTS["madder"] if failed else theme.INK))
        p.drawText(QPointF(r.left() + 14, r.top() + 24), self.word.upper())
        p.setFont(theme.font("text", 10.5, 400))
        p.setPen(theme.qcolor(theme.HAIR))
        p.drawText(QRectF(r.left() + 14, r.top() + 10, r.width() - 28, 18), Qt.AlignRight | Qt.AlignVCenter, "measured, STATUS.md")
        p.setFont(theme.font("text", 27, 600, tabular=True))
        p.setPen(theme.qcolor(theme.INK))
        p.drawText(QRectF(r.left() + 14, r.top() + 28, r.width() - 28, 34), Qt.AlignLeft | Qt.AlignVCenter, self.number)
        p.setFont(theme.font("text", 11, 400))
        p.setPen(theme.qcolor(theme.INK_SOFT))
        p.drawText(QRectF(r.left() + 14, r.top() + 62, r.width() - 28, 30), Qt.AlignLeft | Qt.TextWordWrap, self.label)
        if self.live:
            p.setFont(theme.font("text", 11, 300, tabular=True))
            p.setPen(theme.qcolor(theme.HAIR))
            p.drawText(QRectF(r.left() + 14, r.bottom() - 18, r.width() - 28, 14), Qt.AlignLeft | Qt.AlignVCenter, self.live)
        p.end()


class PlayButton(QWidget):
    toggled = Signal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(40, 40)
        self.playing = False
        self.setCursor(Qt.PointingHandCursor)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setToolTip("Play / pause  (space)")

    def set_playing(self, on: bool):
        self.playing = on
        self.update()

    def mousePressEvent(self, e):
        self.playing = not self.playing
        self.toggled.emit(self.playing)
        self.update()

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        r = QRectF(self.rect()).adjusted(2, 2, -2, -2)
        p.setPen(Qt.NoPen)
        p.setBrush(theme.qcolor(theme.INK if self.isEnabled() else theme.STONE))
        p.drawEllipse(r)
        p.setBrush(theme.qcolor(theme.PAPER))
        c = r.center()
        if self.playing:
            p.drawRoundedRect(QRectF(c.x() - 6, c.y() - 7, 4, 14), 1, 1)
            p.drawRoundedRect(QRectF(c.x() + 2, c.y() - 7, 4, 14), 1, 1)
        else:
            tri = QPainterPath(QPointF(c.x() - 4, c.y() - 7.5))
            tri.lineTo(QPointF(c.x() + 7.5, c.y()))
            tri.lineTo(QPointF(c.x() - 4, c.y() + 7.5))
            tri.closeSubpath()
            p.drawPath(tri)
        if self.hasFocus():
            p.setBrush(Qt.NoBrush)
            p.setPen(QPen(theme.qcolor(theme.PIGMENTS["ochre"]), 2))
            p.drawEllipse(r.adjusted(-1, -1, 1, 1))
        p.end()


class TimeBar(QWidget):
    """The whole time range on screen at once, with the playhead in the scene's pigment."""

    scrubbed = Signal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumWidth(320)
        self.setFixedHeight(56)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.t0, self.t1, self.t = 0.0, 1.0, 0.0
        self.unit = ""
        self.pigment = theme.INK
        self.note = ""
        self.setCursor(Qt.PointingHandCursor)

    def set_range(self, t0, t1, unit="", pigment=theme.INK):
        self.t0, self.t1, self.unit, self.pigment = float(t0), float(t1), unit, pigment
        self.t = self.t0
        self.update()

    def set_t(self, t):
        self.t = float(t)
        self.update()

    def _track(self) -> QRectF:
        return QRectF(16, 30, self.width() - 32, 2)

    def _t_at(self, x):
        tr = self._track()
        f = min(max((x - tr.left()) / tr.width(), 0.0), 1.0)
        return self.t0 + f * (self.t1 - self.t0)

    def mousePressEvent(self, e):
        self.scrubbed.emit(self._t_at(e.position().x()))

    def mouseMoveEvent(self, e):
        if e.buttons() & Qt.LeftButton:
            self.scrubbed.emit(self._t_at(e.position().x()))

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        tr = self._track()
        p.setPen(Qt.NoPen)
        p.setBrush(theme.qcolor(theme.INK, 0.35))
        p.drawRect(tr)
        span = self.t1 - self.t0
        step = _nice_step(span / 8)
        p.setFont(theme.font("text", 11, 500, tabular=True))
        v = math.ceil(self.t0 / step) * step
        while v <= self.t1 + 1e-9:
            x = tr.left() + (v - self.t0) / span * tr.width()
            p.setPen(QPen(theme.qcolor(theme.INK, 0.45), 1))
            p.drawLine(QPointF(x, tr.top() - 4), QPointF(x, tr.top() + 6))
            p.setPen(theme.qcolor(theme.INK_SOFT))
            s = _fmt_t(v)
            p.drawText(QPointF(x - p.fontMetrics().horizontalAdvance(s) / 2, tr.top() + 22), s)
            v += step
        f = (self.t - self.t0) / span if span else 0
        x = tr.left() + f * tr.width()
        p.setPen(Qt.NoPen)
        p.setBrush(theme.qcolor(self.pigment))
        p.drawRect(QRectF(tr.left(), tr.top() - 1, x - tr.left(), 4))
        p.drawPath(blot(QPointF(x, tr.top() + 1), 7.5, seed=91))
        p.setFont(theme.font("text", 13, 600, tabular=True))
        p.setPen(theme.qcolor(theme.INK))
        label = f"t = {_fmt_t(self.t, 2)}{(' ' + self.unit) if self.unit else ''}"
        lx = min(max(x - 30, 0), self.width() - 120)
        p.drawText(QPointF(lx, 15), label)
        if self.note:
            p.setFont(theme.font("text", 11, 400))
            p.setPen(theme.qcolor(theme.HAIR))
            p.drawText(QRectF(lx + 92, 2, self.width() - lx - 96, 16), Qt.AlignLeft | Qt.AlignVCenter, self.note)
        p.end()


class ReBar(QWidget):
    """Reynolds number. Discrete: every trained value on screen. Continuous: the held-out band hatched."""

    changed = Signal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumWidth(320)
        self.setFixedHeight(56)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.mode = "discrete"
        self.values: List[float] = [100, 400, 1000]
        self.lo, self.hi = 100.0, 1000.0
        self.Re = 1000.0
        self.holdout = None
        self.refs = [100, 400, 1000]
        self.pigment = theme.INK
        self.setCursor(Qt.PointingHandCursor)

    def configure(self, mode, values=None, lo=100, hi=1000, holdout=None, pigment=theme.INK, Re=1000):
        self.mode, self.lo, self.hi, self.holdout, self.pigment, self.Re = mode, float(lo), float(hi), holdout, pigment, float(Re)
        if values:
            self.values = [float(v) for v in values]
        self.update()

    def _track(self) -> QRectF:
        return QRectF(78, 30, self.width() - 104, 2)

    def _x(self, Re):
        tr = self._track()
        f = (math.log10(Re) - math.log10(self.lo)) / (math.log10(self.hi) - math.log10(self.lo))
        return tr.left() + f * tr.width()

    def _Re_at(self, x):
        tr = self._track()
        f = min(max((x - tr.left()) / tr.width(), 0.0), 1.0)
        return 10 ** (math.log10(self.lo) + f * (math.log10(self.hi) - math.log10(self.lo)))

    def _pick(self, x):
        if self.mode == "discrete":
            Re = min(self.values, key=lambda v: abs(self._x(v) - x))
        else:
            Re = round(self._Re_at(x) / 5) * 5
            for r in self.refs:  # snap onto a reference value when close, so its reference print appears
                if abs(self._x(r) - x) < 7:
                    Re = r
        if Re != self.Re:
            self.Re = float(Re)
            self.changed.emit(self.Re)
            self.update()

    def mousePressEvent(self, e):
        self._pick(e.position().x())

    def mouseMoveEvent(self, e):
        if e.buttons() & Qt.LeftButton and self.mode == "continuous":
            self._pick(e.position().x())

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        tr = self._track()
        p.setFont(theme.font("text", 13, 600))
        p.setPen(theme.qcolor(theme.INK))
        p.drawText(QPointF(0, 35), "Re")
        if self.mode == "continuous":
            p.setFont(theme.font("text", 11, 400))
            p.setPen(theme.qcolor(theme.HAIR))
            p.drawText(QPointF(0, 52), "lid speed")
        p.setFont(theme.font("text", 11, 500, tabular=True))
        p.setPen(Qt.NoPen)
        p.setBrush(theme.qcolor(theme.INK, 0.35))
        p.drawRect(tr)
        if self.mode == "continuous" and self.holdout:
            a, b = self._x(self.holdout[0]), self._x(self.holdout[1])
            band = QRectF(a, tr.top() - 9, b - a, 20)
            p.setBrush(QBrush(theme.qcolor(theme.INK, 0.55), Qt.BDiagPattern))
            p.drawRect(band)
            p.setPen(theme.qcolor(theme.INK_SOFT))
            p.drawText(QPointF(a, tr.top() - 13), "never seen in training")
        marks = self.values if self.mode == "discrete" else self.refs
        for v in marks:
            x = self._x(v)
            on = abs(v - self.Re) < 1e-6
            p.setPen(QPen(theme.qcolor(theme.INK, 0.5), 1))
            p.drawLine(QPointF(x, tr.top() - 4), QPointF(x, tr.top() + 6))
            p.setFont(theme.font("text", 12 if on else 11, 650 if on else 500, tabular=True))
            p.setPen(theme.qcolor(theme.INK if on else theme.INK_SOFT))
            s = f"{v:.0f}"
            p.drawText(QPointF(x - p.fontMetrics().horizontalAdvance(s) / 2, tr.top() + 22), s)
        x = self._x(self.Re)
        p.setPen(Qt.NoPen)
        p.setBrush(theme.qcolor(self.pigment))
        p.drawPath(blot(QPointF(x, tr.top() + 1), 8, seed=57))
        if self.mode == "continuous":
            p.setFont(theme.font("text", 13, 600, tabular=True))
            p.setPen(theme.qcolor(theme.INK))
            p.drawText(QPointF(26, 35), f"{self.Re:.0f}")
        p.end()


class ViewChips(QWidget):
    chosen = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(56)
        self.views: List[str] = []
        self.active = "pigment"
        self.evidence: Optional[str] = None
        self.pigment = theme.INK
        self.setCursor(Qt.PointingHandCursor)

    def configure(self, views, active, pigment, evidence=None):
        self.views, self.active, self.pigment, self.evidence = list(views), active, pigment, evidence
        self.setFixedWidth(self._width())
        self.update()

    def _boxes(self):
        boxes, x = [], 0.0
        for v in self.views:
            f = theme.font("text", 13, 650 if v == self.active else 500)
            from PySide6.QtGui import QFontMetricsF

            w = QFontMetricsF(f).horizontalAdvance(theme.VIEWS.get(v, {"label": v})["label"]) + 22
            boxes.append((v, QRectF(x, 14, w, 30)))
            x += w + 4
        return boxes

    def _width(self):
        b = self._boxes()
        return int(b[-1][1].right() + 4) if b else 10

    def mousePressEvent(self, e):
        for v, r in self._boxes():
            if r.contains(e.position()):
                self.chosen.emit(v)

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        for v, r in self._boxes():
            on = v == self.active
            p.setFont(theme.font("text", 13, 650 if on else 500))
            p.setPen(theme.qcolor(theme.INK if on else theme.INK_SOFT))
            label = theme.VIEWS.get(v, {"label": v})["label"]
            p.drawText(r, Qt.AlignCenter, label)
            if on:
                p.setPen(Qt.NoPen)
                p.setBrush(theme.qcolor(self.pigment))
                p.drawRect(QRectF(r.left() + 10, r.bottom() - 2, r.width() - 20, 2.5))
            if v == self.evidence and not on:
                p.setPen(Qt.NoPen)
                p.setBrush(theme.qcolor(theme.PIGMENTS["madder"]))
                p.drawEllipse(QPointF(r.right() - 6, r.top() + 6), 2.6, 2.6)
        p.end()


def _nice_step(x):
    e = 10 ** math.floor(math.log10(max(x, 1e-12)))
    for m in (1, 2, 2.5, 5, 10):
        if m * e >= x:
            return m * e
    return 10 * e


def _fmt_t(v, d=None):
    if d is not None:
        return f"{v:.{d}f}"
    return f"{v:g}" if abs(v - round(v)) > 1e-9 or abs(v) < 1e4 else f"{v:.0f}"
