"""Controls in the bath's vocabulary. Every choice stays on screen; the active one is struck forward.

Every control takes keyboard focus (Tab) and arrow keys; its focus ring is an ink rule. State changes ease out over
theme.MOTION_MS through Tween: the rail's strike, the chip underline, the playheads and the stamp's impression.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional

import numpy as np
from PySide6.QtCore import QAbstractAnimation, QEasingCurve, QPointF, QRectF, Qt, QVariantAnimation, Signal
from PySide6.QtGui import QBrush, QFont, QFontMetricsF, QImage, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QSizePolicy, QWidget

from viewer import materials, theme
from viewer.scenes import SCENES


NBSP = "\u00a0"


class Tween:
    """A float that eases out to its target and repaints its widget on every step."""

    def __init__(self, widget: QWidget, ms: int = theme.MOTION_MS):
        self.w = widget
        self.v: Optional[float] = None
        self.anim = QVariantAnimation(widget)
        self.anim.setDuration(ms)
        self.anim.setEasingCurve(QEasingCurve.OutExpo)
        self.anim.valueChanged.connect(self._on)

    def _on(self, v):
        self.v = float(v)
        self.w.update()

    def to(self, target: float, animate: bool = True):
        target = float(target)
        running = self.anim.state() == QAbstractAnimation.Running
        if self.v is None or not animate:
            self.anim.stop()
            self.v = target
            self.w.update()
            return
        if running and abs(float(self.anim.endValue()) - target) < 1e-12:
            return
        if not running and abs(self.v - target) < 1e-12:
            return
        self.anim.stop()
        self.anim.setStartValue(self.v)
        self.anim.setEndValue(target)
        self.anim.start()

    def get(self, default: float = 0.0) -> float:
        return default if self.v is None else self.v


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


def balanced_lines(text: str, font, width: float, max_lines: int = 3) -> List[str]:
    """Wrap into the fewest lines that fit, then even them out so no line is left with a lone word."""
    fm = QFontMetricsF(font)
    words = text.split(" ")
    for n in range(1, max_lines + 1):
        # binary search the narrowest width that still wraps into n lines
        lo, hi = fm.horizontalAdvance(max(words, key=fm.horizontalAdvance)), width
        if lo > width:
            break
        best = None
        for _ in range(18):
            mid = (lo + hi) / 2
            lines = _greedy(words, fm, mid)
            if len(lines) <= n:
                best, hi = lines, mid
            else:
                lo = mid
        if best is not None and len(best) <= n and all(fm.horizontalAdvance(s) <= width for s in best):
            return best
    return _greedy(words, fm, width)


def _greedy(words, fm, width) -> List[str]:
    lines, cur = [], ""
    for w in words:
        trial = (cur + " " + w).strip()
        if cur and fm.horizontalAdvance(trial) > width:
            lines.append(cur)
            cur = w
        else:
            cur = trial
    if cur:
        lines.append(cur)
    return lines


class KeyboardFocus:
    """Mixin: remember whether focus arrived from the keyboard, so rings show for keyboard users only."""

    _kbd = False

    def focusInEvent(self, e):
        self._kbd = e.reason() in (Qt.TabFocusReason, Qt.BacktabFocusReason)
        self.update()
        super().focusInEvent(e)

    def focusOutEvent(self, e):
        self._kbd = False
        self.update()
        super().focusOutEvent(e)

    def ring_visible(self) -> bool:
        return self.hasFocus() and self._kbd


def focus_ring(p: QPainter, r: QRectF, radius: float = 4.0):
    p.save()
    p.setBrush(Qt.NoBrush)
    p.setPen(QPen(theme.qcolor(theme.INK), 1.5))
    p.drawRoundedRect(r, radius, radius)
    p.restore()


class SceneRail(KeyboardFocus, QWidget):
    chosen = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(78)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setAccessibleName("Scenes")
        self.active = SCENES[0].sid
        self.ready: Dict[str, str] = {s.sid: "waiting" for s in SCENES}
        self.available = {s.sid for s in SCENES}
        self.hover = None
        self.strike = Tween(self)

    def set_active(self, sid: str, animate: bool = True):
        self.active = sid
        self.strike.to(next(i for i, s in enumerate(SCENES) if s.sid == sid), animate)
        self.update()

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

    def keyPressEvent(self, e):
        if e.key() in (Qt.Key_Left, Qt.Key_Right):
            ids = [s.sid for s in SCENES if s.sid in self.available]
            if self.active in ids:
                i = (ids.index(self.active) + (1 if e.key() == Qt.Key_Right else -1)) % len(ids)
                self.chosen.emit(ids[i])
            e.accept()
            return
        super().keyPressEvent(e)

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        cells = self._cells()
        for i, (s, r) in enumerate(zip(SCENES, cells)):
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
            p.setFont(theme.font("text", 14, 650 if on else 500))
            p.setPen(theme.qcolor(theme.INK if on else theme.INK_SOFT))
            p.drawText(QRectF(x, r.top() + 11, r.width() - 60, 20), Qt.AlignLeft | Qt.AlignVCenter, s.short)
            p.setFont(theme.font("text", theme.SMALL, 500))
            word = s.verdict if state in ("ready", "waiting") or on else ("compiling" if state == "warming" else state)
            p.setPen(theme.qcolor(theme.INK_SOFT if s.verdict != "failed" else theme.MADDER_TEXT))
            p.drawText(QRectF(x, r.top() + 32, r.width() - 60, 18), Qt.AlignLeft | Qt.AlignVCenter, word)
            p.setOpacity(1.0)
        # the strike: one rule in the active pigment that slides to the chosen drop
        if cells:
            k = self.strike.get(0.0)
            i0 = int(math.floor(k))
            f = k - i0
            w = cells[0].width()
            x = k * w
            colour = next(s.pigment for s in SCENES if s.sid == self.active)
            p.setPen(Qt.NoPen)
            p.setBrush(theme.qcolor(colour))
            p.drawRect(QRectF(x + 10, self.height() - 6, w - 20, 3))
            if self.ring_visible():
                focus_ring(p, QRectF(x + 4, 3, w - 8, self.height() - 12), 6)
        p.setPen(QPen(theme.qcolor(theme.INK, 0.25), 1))
        p.drawLine(QPointF(0, self.height() - 1), QPointF(self.width(), self.height() - 1))
        p.end()


class VerdictStamp(QWidget):
    """The verdict as an ink stamp pressed on the tray's lip.

    The impression (double rule, the word in Bodoni capitals, the measured number, what it measures, and where the
    number comes from) is drawn in the verdict's ink, masked by an uneven ink density, and set at a slight angle.
    The number derived live from the current frame is not stamped: it sits in hairline on the tray's bottom lip.
    """

    W, H = 316, 120
    IMP_W, IMP_H = 296, 110
    ANGLE = -1.5

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(self.W, self.H)
        self.word = ""
        self.number = ""
        self.label = ""
        self.pigment = theme.INK
        self._imp: Optional[QImage] = None
        self.press = Tween(self, 220)
        self.press.to(1.0, animate=False)

    def set(self, word: str, number: str, label: str, pigment: str):
        changed = (word, number, label) != (self.word, self.number, self.label)
        self.word, self.number, self.label, self.pigment = word, number, label, pigment
        self._imp = None
        if changed:
            self.press.v = 0.0
            self.press.to(1.0)
        self.update()

    def ink(self) -> str:
        return theme.MADDER_TEXT if self.word == "failed" else theme.INK

    def _impression(self, dpr: float) -> QImage:
        w, h = int(self.IMP_W * dpr), int(self.IMP_H * dpr)
        img = QImage(w, h, QImage.Format_ARGB32_Premultiplied)
        img.setDevicePixelRatio(dpr)
        img.fill(Qt.transparent)
        p = QPainter(img)
        p.setRenderHint(QPainter.Antialiasing)
        p.setRenderHint(QPainter.TextAntialiasing)
        ink = theme.qcolor(self.ink())
        r = QRectF(1.5, 1.5, self.IMP_W - 3, self.IMP_H - 3)
        p.setBrush(Qt.NoBrush)
        p.setPen(QPen(ink, 2.4, Qt.SolidLine, Qt.SquareCap, Qt.MiterJoin))
        p.drawRect(r)
        p.setPen(QPen(ink, 0.9, Qt.SolidLine, Qt.SquareCap, Qt.MiterJoin))
        p.drawRect(r.adjusted(4.5, 4.5, -4.5, -4.5))
        x0, x1 = r.left() + 15, r.right() - 15
        f = theme.font("display", 18, 650, opsz=18)
        f.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, 1.6)
        p.setFont(f)
        p.setPen(ink)
        p.drawText(QPointF(x0, r.top() + 31), self.word.upper())
        p.setFont(theme.font("text", 30, 600, tabular=True))
        p.drawText(QRectF(x0, r.top() + 8, x1 - x0, 34), Qt.AlignRight | Qt.AlignVCenter, self.number)
        p.setPen(QPen(ink, 0.8))
        p.drawLine(QPointF(x0, r.top() + 46), QPointF(x1, r.top() + 46))
        lf = theme.font("text", theme.SMALL, 450)
        p.setFont(lf)
        lines = balanced_lines(self.label.replace(" %", NBSP + "%"), lf, x1 - x0, 2)
        for i, s in enumerate(lines[:2]):
            p.drawText(QPointF(x0, r.top() + 63 + i * 16.5), s)
        p.setFont(theme.font("text", theme.SMALL, 650))
        p.drawText(QPointF(x0, r.top() + 97), "measured, STATUS.md")
        p.setCompositionMode(QPainter.CompositionMode_DestinationIn)
        p.drawImage(QRectF(0, 0, self.IMP_W, self.IMP_H), materials.ink_texture(w, h))
        p.end()
        return img

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setRenderHint(QPainter.SmoothPixmapTransform)
        dpr = self.devicePixelRatioF()
        if self._imp is None or abs(self._imp.devicePixelRatio() - dpr) > 1e-3:
            self._imp = self._impression(dpr)
        k = self.press.get(1.0)
        cx, cy = self.W / 2, 6 + self.IMP_H / 2
        p.save()
        p.translate(cx, cy)
        p.rotate(self.ANGLE)
        s = 1.0 + 0.05 * (1 - k)
        p.scale(s, s)
        p.setOpacity(0.25 + 0.75 * k)
        p.drawImage(QRectF(-self.IMP_W / 2, -self.IMP_H / 2, self.IMP_W, self.IMP_H), self._imp)
        p.restore()
        p.end()


class PlayButton(KeyboardFocus, QWidget):
    toggled = Signal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(44, 44)
        self.playing = False
        self.setCursor(Qt.PointingHandCursor)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setAccessibleName("Play")
        self.setToolTip("Play / pause  (space)")

    def set_playing(self, on: bool):
        self.playing = on
        self.setAccessibleName("Pause" if on else "Play")
        self.update()

    def mousePressEvent(self, e):
        self.playing = not self.playing
        self.toggled.emit(self.playing)
        self.update()

    def keyPressEvent(self, e):
        if e.key() in (Qt.Key_Return, Qt.Key_Enter):
            self.mousePressEvent(e)
            return
        super().keyPressEvent(e)

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        r = QRectF(self.rect()).adjusted(4, 4, -4, -4)
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
        if self.ring_visible():
            p.setBrush(Qt.NoBrush)
            p.setPen(QPen(theme.qcolor(theme.INK), 1.5))
            p.drawEllipse(r.adjusted(-3, -3, 3, 3))
        p.end()


class TimeBar(KeyboardFocus, QWidget):
    """The whole time range on screen at once, with the playhead in the scene's pigment."""

    scrubbed = Signal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumWidth(320)
        self.setFixedHeight(58)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setAccessibleName("Time")
        self.t0, self.t1, self.t = 0.0, 1.0, 0.0
        self.unit = ""
        self.pigment = theme.INK
        self.head = Tween(self)
        self.setCursor(Qt.PointingHandCursor)

    def set_range(self, t0, t1, unit="", pigment=theme.INK):
        self.t0, self.t1, self.unit, self.pigment = float(t0), float(t1), unit, pigment
        self.t = self.t0
        self.head.to(0.0, animate=False)
        self.update()

    def set_t(self, t, animate: bool = False):
        """animate=True for a user's jump (click, keys); playback and loop wraps snap."""
        self.t = float(t)
        span = self.t1 - self.t0
        f = (self.t - self.t0) / span if span else 0.0
        self.head.to(f, animate=animate and abs(f - self.head.get(f)) > 0.04)

    def _track(self) -> QRectF:
        return QRectF(16, 32, self.width() - 32, 2)

    def _t_at(self, x):
        tr = self._track()
        f = min(max((x - tr.left()) / tr.width(), 0.0), 1.0)
        return self.t0 + f * (self.t1 - self.t0)

    def mousePressEvent(self, e):
        self.scrubbed.emit(self._t_at(e.position().x()))

    def mouseMoveEvent(self, e):
        if e.buttons() & Qt.LeftButton:
            self.scrubbed.emit(self._t_at(e.position().x()))

    def keyPressEvent(self, e):
        span = self.t1 - self.t0
        step = span / (10 if e.modifiers() & Qt.ShiftModifier else 50)
        k = e.key()
        if k in (Qt.Key_Left, Qt.Key_Right, Qt.Key_Home, Qt.Key_End):
            t = {Qt.Key_Left: self.t - step, Qt.Key_Right: self.t + step, Qt.Key_Home: self.t0, Qt.Key_End: self.t1}[k]
            self.scrubbed.emit(min(max(t, self.t0), self.t1))
            e.accept()
            return
        super().keyPressEvent(e)

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        tr = self._track()
        p.setPen(Qt.NoPen)
        p.setBrush(theme.qcolor(theme.INK, 0.35))
        p.drawRect(tr)
        span = self.t1 - self.t0
        step = _nice_step(span / 8)
        p.setFont(theme.font("text", theme.SMALL, 500, tabular=True))
        v = math.ceil(self.t0 / step) * step
        while v <= self.t1 + 1e-9:
            x = tr.left() + (v - self.t0) / span * tr.width()
            p.setPen(QPen(theme.qcolor(theme.INK, 0.45), 1))
            p.drawLine(QPointF(x, tr.top() - 4), QPointF(x, tr.top() + 6))
            p.setPen(theme.qcolor(theme.INK_SOFT))
            s = _fmt_t(v)
            p.drawText(QPointF(x - p.fontMetrics().horizontalAdvance(s) / 2, tr.top() + 23), s)
            v += step
        x = tr.left() + self.head.get(0.0) * tr.width()
        p.setPen(Qt.NoPen)
        p.setBrush(theme.qcolor(self.pigment))
        p.drawRect(QRectF(tr.left(), tr.top() - 1, x - tr.left(), 4))
        p.drawPath(blot(QPointF(x, tr.top() + 1), 7.5, seed=91))
        p.setFont(theme.font("text", 14, 600, tabular=True))
        p.setPen(theme.qcolor(theme.INK))
        label = f"t = {_fmt_t(self.t, 2)}{(' ' + self.unit) if self.unit else ''}"
        lw = p.fontMetrics().horizontalAdvance(label)
        lx = min(max(x - lw / 2, 0), self.width() - lw)
        p.drawText(QPointF(lx, 16), label)
        if self.ring_visible():
            focus_ring(p, QRectF(self.rect()).adjusted(1, 1, -1, -1))
        p.end()


class ReBar(KeyboardFocus, QWidget):
    """Reynolds number. Discrete: every trained value on screen. Continuous: the held-out band hatched."""

    changed = Signal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumWidth(320)
        self.setFixedHeight(58)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setAccessibleName("Reynolds number")
        self.mode = "discrete"
        self.values: List[float] = [100, 400, 1000]
        self.lo, self.hi = 100.0, 1000.0
        self.Re = 1000.0
        self.holdout = None
        self.refs = [100, 400, 1000]
        self.pigment = theme.INK
        self.mark = Tween(self)
        self.setCursor(Qt.PointingHandCursor)

    def configure(self, mode, values=None, lo=100, hi=1000, holdout=None, pigment=theme.INK, Re=1000):
        self.mode, self.lo, self.hi, self.holdout, self.pigment, self.Re = mode, float(lo), float(hi), holdout, pigment, float(Re)
        if values:
            self.values = [float(v) for v in values]
        self.mark.to(self._f(self.Re), animate=False)
        self.update()

    def set_Re(self, Re: float):
        self.Re = float(Re)
        f = self._f(self.Re)
        self.mark.to(f, animate=abs(f - self.mark.get(f)) > 0.04)

    def _track(self) -> QRectF:
        return QRectF(96, 32, self.width() - 122, 2)

    def _f(self, Re):
        return (math.log10(max(Re, 1e-9)) - math.log10(self.lo)) / (math.log10(self.hi) - math.log10(self.lo))

    def _x(self, Re):
        tr = self._track()
        return tr.left() + self._f(Re) * tr.width()

    def _Re_at(self, x):
        tr = self._track()
        f = min(max((x - tr.left()) / tr.width(), 0.0), 1.0)
        return 10 ** (math.log10(self.lo) + f * (math.log10(self.hi) - math.log10(self.lo)))

    def _emit(self, Re):
        if Re != self.Re:
            self.set_Re(Re)
            self.changed.emit(self.Re)

    def _pick(self, x):
        if self.mode == "discrete":
            Re = min(self.values, key=lambda v: abs(self._x(v) - x))
        else:
            Re = round(self._Re_at(x) / 5) * 5
            for r in self.refs:  # snap onto a reference value when close, so its reference print appears
                if abs(self._x(r) - x) < 7:
                    Re = r
        self._emit(float(Re))

    def mousePressEvent(self, e):
        self._pick(e.position().x())

    def mouseMoveEvent(self, e):
        if e.buttons() & Qt.LeftButton and self.mode == "continuous":
            self._pick(e.position().x())

    def keyPressEvent(self, e):
        k = e.key()
        if k not in (Qt.Key_Left, Qt.Key_Right):
            return super().keyPressEvent(e)
        d = 1 if k == Qt.Key_Right else -1
        if self.mode == "discrete":
            i = min(range(len(self.values)), key=lambda j: abs(self.values[j] - self.Re))
            self._emit(self.values[min(max(i + d, 0), len(self.values) - 1)])
        else:
            step = 50 if e.modifiers() & Qt.ShiftModifier else 10
            self._emit(float(min(max(self.Re + d * step, self.lo), self.hi)))
        e.accept()

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        tr = self._track()
        p.setFont(theme.font("text", 14, 600, tabular=True))
        p.setPen(theme.qcolor(theme.INK))
        p.drawText(QPointF(4, 37), "Re" if self.mode == "discrete" else f"Re {self.Re:.0f}")
        if self.mode == "continuous":
            p.setFont(theme.font("text", theme.SMALL, 400))
            p.setPen(theme.qcolor(theme.INK_SOFT))
            p.drawText(QPointF(4, 54), "lid speed")
        p.setPen(Qt.NoPen)
        p.setBrush(theme.qcolor(theme.INK, 0.35))
        p.drawRect(tr)
        p.setFont(theme.font("text", theme.SMALL, 500, tabular=True))
        if self.mode == "continuous" and self.holdout:
            a, b = self._x(self.holdout[0]), self._x(self.holdout[1])
            band = QRectF(a, tr.top() - 9, b - a, 20)
            p.setBrush(QBrush(theme.qcolor(theme.INK, 0.55), Qt.BDiagPattern))
            p.drawRect(band)
            p.setPen(theme.qcolor(theme.INK_SOFT))
            p.drawText(QPointF(a, tr.top() - 14), "never seen in training")
        marks = self.values if self.mode == "discrete" else self.refs
        for v in marks:
            x = self._x(v)
            on = abs(v - self.Re) < 1e-6
            p.setPen(QPen(theme.qcolor(theme.INK, 0.5), 1))
            p.drawLine(QPointF(x, tr.top() - 4), QPointF(x, tr.top() + 6))
            p.setFont(theme.font("text", 14 if on else theme.SMALL, 650 if on else 500, tabular=True))
            p.setPen(theme.qcolor(theme.INK if on else theme.INK_SOFT))
            s = f"{v:.0f}"
            p.drawText(QPointF(x - p.fontMetrics().horizontalAdvance(s) / 2, tr.top() + 23), s)
        x = tr.left() + self.mark.get(self._f(self.Re)) * tr.width()
        p.setPen(Qt.NoPen)
        p.setBrush(theme.qcolor(self.pigment))
        p.drawPath(blot(QPointF(x, tr.top() + 1), 8, seed=57))
        if self.ring_visible():
            focus_ring(p, QRectF(self.rect()).adjusted(1, 1, -1, -1))
        p.end()


class ViewChips(KeyboardFocus, QWidget):
    chosen = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(58)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setAccessibleName("Views")
        self.views: List[str] = []
        self.active = "pigment"
        self.evidence: Optional[str] = None
        self.pigment = theme.INK
        self.ux = Tween(self)
        self.uw = Tween(self)
        self.setCursor(Qt.PointingHandCursor)

    def configure(self, views, active, pigment, evidence=None, animate=True):
        same = list(views) == self.views
        self.views, self.active, self.pigment, self.evidence = list(views), active, pigment, evidence
        self.setFixedWidth(self._width())
        box = dict(self._boxes()).get(active)
        if box is not None:
            self.ux.to(box.left() + 10, animate and same)
            self.uw.to(box.width() - 20, animate and same)
        self.update()

    def _boxes(self):
        boxes, x = [], 0.0
        for v in self.views:
            f = theme.font("text", 14, 650)  # widths at the active weight, so nothing shifts when a chip is chosen
            w = QFontMetricsF(f).horizontalAdvance(theme.VIEWS.get(v, {"label": v})["label"]) + 24
            boxes.append((v, QRectF(x, 14, w, 32)))
            x += w + 4
        return boxes

    def _width(self):
        b = self._boxes()
        return int(b[-1][1].right() + 4) if b else 10

    def mousePressEvent(self, e):
        for v, r in self._boxes():
            if r.contains(e.position()):
                self.chosen.emit(v)

    def keyPressEvent(self, e):
        if e.key() in (Qt.Key_Left, Qt.Key_Right) and self.views:
            i = self.views.index(self.active) if self.active in self.views else 0
            self.chosen.emit(self.views[(i + (1 if e.key() == Qt.Key_Right else -1)) % len(self.views)])
            e.accept()
            return
        super().keyPressEvent(e)

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        for v, r in self._boxes():
            on = v == self.active
            p.setFont(theme.font("text", 14, 650 if on else 500))
            p.setPen(theme.qcolor(theme.INK if on else theme.INK_SOFT))
            label = theme.VIEWS.get(v, {"label": v})["label"]
            p.drawText(r, Qt.AlignCenter, label)
            if v == self.evidence and not on:
                p.setPen(Qt.NoPen)
                p.setBrush(theme.qcolor(theme.MADDER_TEXT))
                p.drawEllipse(QPointF(r.right() - 6, r.top() + 6), 3.0, 3.0)
            if on and self.ring_visible():
                focus_ring(p, r.adjusted(1, 0, -1, 2))
        p.setPen(Qt.NoPen)
        p.setBrush(theme.qcolor(self.pigment))
        p.drawRect(QRectF(self.ux.get(0.0), 14 + 32 - 2, self.uw.get(0.0), 2.5))
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
