"""The bath: the water in the tray, where the network's current carries the pigment (2D scenes).

Painted with QPainter. The field image and the pigment raster are both produced off the UI thread; this widget
only composes them, draws the obstacles, the measured call-outs and the states, and turns clicks and drags into
pigment. The tray itself (lip, shadow, wet edge) is painted by panel.BathPanel around this widget.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QImage, QLinearGradient, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QSizePolicy, QWidget

from viewer import theme


class Bath2D(QWidget):
    dropped = Signal(object)       # (M, 2) points in domain coordinates
    raster_size = Signal(int, int)  # pigment raster resolution wanted by the painter

    MARGIN = 10

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMouseTracking(True)
        self.setCursor(Qt.CrossCursor)
        self.domain: List[float] = [0, 1, 0, 1]
        self.field_img: Optional[QImage] = None
        self.pigment_img: Optional[QImage] = None
        self.pigment_opacity = 1.0
        self.cylinder = None
        self.sensors: Optional[np.ndarray] = None
        self.state = "connecting"  # connecting | compiling | live | offline
        self.state_text = ""
        self.legend = None  # (lut, lo, hi, label)
        self.callouts: List[dict] = []  # {"x", "y", "text", "side"}: measured values pinned to a place in the flow
        self.walls = None  # (x, h): a vessel wall at y = +-h(x), drawn as an ink line
        self._drag: List[QPointF] = []
        self._press = None
        self._sweep = 0.0
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(40)

    # ---- geometry ----
    LEGEND = 28

    def domain_rect(self) -> QRectF:
        """The flow domain, as large as the widget allows at its true aspect ratio."""
        m = self.MARGIN
        avail = QRectF(self.rect()).adjusted(m, m, -m, -m - self.LEGEND)
        x0, x1, y0, y1 = self.domain[:4]
        aspect = (x1 - x0) / max(y1 - y0, 1e-9)
        w, h = avail.width(), avail.height()
        if w / max(h, 1) > aspect:
            nw = h * aspect
            return QRectF(avail.left() + (w - nw) / 2, avail.top(), nw, h)
        nh = w / aspect
        return QRectF(avail.left(), avail.top() + (h - nh) / 2, w, nh)

    def to_domain(self, p: QPointF) -> Optional[np.ndarray]:
        r = self.domain_rect()
        if not r.contains(p):
            return None
        x0, x1, y0, y1 = self.domain[:4]
        x = x0 + (p.x() - r.left()) / r.width() * (x1 - x0)
        y = y1 - (p.y() - r.top()) / r.height() * (y1 - y0)
        return np.array([x, y], np.float32)

    def to_px(self, x: float, y: float) -> QPointF:
        r = self.domain_rect()
        x0, x1, y0, y1 = self.domain[:4]
        return QPointF(r.left() + (x - x0) / (x1 - x0) * r.width(), r.top() + (y1 - y) / (y1 - y0) * r.height())

    def resizeEvent(self, e):
        r = self.domain_rect()
        dpr = self.devicePixelRatioF()
        self.raster_size.emit(max(8, int(r.width() * dpr / 2)), max(8, int(r.height() * dpr / 2)))
        super().resizeEvent(e)

    # ---- inputs ----
    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            self._press = e.position()
            self._drag = [e.position()]

    def mouseMoveEvent(self, e):
        if self._press is not None:
            self._drag.append(e.position())
            self.update()

    def mouseReleaseEvent(self, e):
        if self._press is None:
            return
        path = [self.to_domain(p) for p in self._drag]
        path = [p for p in path if p is not None]
        self._press = None
        self._drag = []
        if not path:
            return
        x0, x1, y0, y1 = self.domain[:4]
        span = np.array([x1 - x0, y1 - y0], np.float32)
        rng = np.random.default_rng()
        length = sum(np.linalg.norm((b - a) / span) for a, b in zip(path[:-1], path[1:]))
        if length < 0.01:  # a drop: a soft disk of pigment
            n = 360
            r = 0.035 * min(span) * np.sqrt(rng.random(n))
            th = rng.random(n) * 2 * np.pi
            pts = path[0] + np.stack([r * np.cos(th), r * np.sin(th)], -1)
        else:  # a comb stroke: pigment laid along the drag
            seg = np.array(path, np.float32)
            d = np.linalg.norm(np.diff(seg, axis=0) / span, axis=1)
            s = np.concatenate([[0], np.cumsum(d)])
            q = np.linspace(0, s[-1], 420)
            pts = np.stack([np.interp(q, s, seg[:, 0]), np.interp(q, s, seg[:, 1])], -1)
            pts = pts + rng.normal(0, 0.004, pts.shape) * span
        lo = np.array([x0, y0], np.float32)
        hi = np.array([x1, y1], np.float32)
        self.dropped.emit(np.clip(pts, lo, hi).astype(np.float32))

    def _tick(self):
        if self.state in ("compiling", "connecting"):
            self._sweep = (self._sweep + 0.012) % 1.0
            self.update()

    # ---- painting ----
    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setRenderHint(QPainter.SmoothPixmapTransform)
        r = self.domain_rect()
        if self.field_img is not None:
            p.drawImage(r, self.field_img)
        if self.pigment_img is not None:
            p.setOpacity(self.pigment_opacity)
            p.drawImage(r, self.pigment_img)
            p.setOpacity(1.0)
        if self.cylinder is not None:
            c = self.to_px(*self.cylinder["center"])
            rr = self.cylinder["radius"] / (self.domain[1] - self.domain[0]) * r.width()
            p.setBrush(theme.qcolor(theme.STONE))
            p.setPen(QPen(theme.qcolor(theme.INK, 0.8), 1.2))
            p.drawEllipse(c, rr, rr)
        if self.walls is not None:
            xs, hs = self.walls
            p.setBrush(Qt.NoBrush)
            p.setPen(QPen(theme.qcolor(theme.INK, 0.85), 1.4))
            for sgn in (1.0, -1.0):
                path = QPainterPath(self.to_px(float(xs[0]), sgn * float(hs[0])))
                for xv, hv in zip(xs[1:], hs[1:]):
                    path.lineTo(self.to_px(float(xv), sgn * float(hv)))
                p.drawPath(path)
        if self.sensors is not None and len(self.sensors):
            p.setPen(QPen(theme.qcolor(theme.INK, 0.85), 1.0))
            p.setBrush(theme.qcolor(theme.PAPER, 0.9))
            for x, y in self.sensors[:600]:
                p.drawEllipse(self.to_px(float(x), float(y)), 2.2, 2.2)
        # domain hairline
        p.setBrush(Qt.NoBrush)
        p.setPen(QPen(theme.qcolor(theme.INK, 0.35), 1.0))
        p.drawRect(r)
        # live comb stroke under the cursor
        if len(self._drag) > 1:
            path = QPainterPath(self._drag[0])
            for q in self._drag[1:]:
                path.lineTo(q)
            p.setPen(QPen(theme.qcolor(theme.INK, 0.45), 1.5, Qt.DashLine))
            p.drawPath(path)
        self._paint_callouts(p, r)
        self._paint_legend(p, r)
        self._paint_state(p, r)
        p.end()

    def _paint_callouts(self, p: QPainter, r: QRectF):
        """A ring on the place, a leader into the water, and the measured value on a slip of paper."""
        if not self.callouts:
            return
        f = theme.font("text", theme.SMALL, 600, tabular=True)
        p.setFont(f)
        fm = p.fontMetrics()
        for c in self.callouts:
            at = self.to_px(c["x"], c["y"])
            at = QPointF(min(max(at.x(), r.left()), r.right()), min(max(at.y(), r.top()), r.bottom()))
            sx = 1 if c.get("side", "right") == "right" else -1
            elbow = QPointF(at.x() + sx * 30, at.y() + 34)
            lines = c["text"].splitlines()
            tw = max(fm.horizontalAdvance(s) for s in lines) + 16
            th = fm.height() * len(lines) + 10
            box = QRectF(elbow.x() if sx > 0 else elbow.x() - tw, elbow.y(), tw, th)
            p.setBrush(Qt.NoBrush)
            p.setPen(QPen(theme.qcolor(theme.PAPER, 0.9), 3.2))  # a pale halo keeps the rule legible on any colour
            p.drawEllipse(at, 9, 9)
            p.drawLine(at + QPointF(sx * 6.4, 6.4), elbow)
            p.setPen(QPen(theme.qcolor(theme.INK), 1.4))
            p.drawEllipse(at, 9, 9)
            p.drawLine(at + QPointF(sx * 6.4, 6.4), elbow)
            p.setPen(QPen(theme.qcolor(theme.INK, 0.55), 0.8))
            p.setBrush(theme.qcolor(theme.PAPER, 0.95))
            p.drawRect(box)
            p.setPen(theme.qcolor(theme.INK))
            for i, s in enumerate(lines):
                p.drawText(QPointF(box.left() + 8, box.top() + 5 + fm.ascent() + i * fm.height()), s)

    def _paint_legend(self, p: QPainter, r: QRectF):
        if self.legend is None:
            return
        lut, lo, hi, label = self.legend
        p.setFont(theme.font("text", theme.SMALL, 500, tabular=True))
        fm = p.fontMetrics()
        lo_s, hi_s = _fmt(lo), _fmt(hi)
        w, h = 180.0, 8.0
        y = r.bottom() + self.LEGEND - 8          # text baseline; the bar sits on it, centred on the figures
        x = r.left()
        p.setPen(theme.qcolor(theme.INK))
        p.drawText(QPointF(x, y), lo_s)          # the scale reads: lo [bar] hi   label
        bx = x + fm.horizontalAdvance(lo_s) + 8
        img = QImage(np.ascontiguousarray(lut[None, :, :]).data, lut.shape[0], 1, 3 * lut.shape[0], QImage.Format_RGB888).copy()
        bar = QRectF(bx, y - fm.ascent() / 2 - h / 2 + 1, w, h)
        p.drawImage(bar, img)
        p.setPen(QPen(theme.qcolor(theme.INK, 0.5), 0.8))
        p.setBrush(Qt.NoBrush)
        p.drawRect(bar)
        x2 = bar.right() + 8
        p.setPen(theme.qcolor(theme.INK))
        p.drawText(QPointF(x2, y), hi_s)
        p.setPen(theme.qcolor(theme.INK_SOFT))
        p.drawText(QPointF(x2 + fm.horizontalAdvance(hi_s) + 16, y), label)

    def _paint_state(self, p: QPainter, r: QRectF):
        if self.state == "live":
            return
        if self.state == "untrained":  # a quiet slip in the corner: the scan below is real data, not a placeholder
            p.setFont(theme.font("text", theme.SMALL, 600))
            text = "network not trained yet: this is the scan it will learn from"
            fm = p.fontMetrics()
            box = QRectF(r.left() + 8, r.top() + 8, fm.horizontalAdvance(text) + 16, fm.height() + 8)
            p.setBrush(theme.qcolor(theme.PAPER, 0.94))
            p.setPen(QPen(theme.qcolor(theme.INK, 0.4), 0.8))
            p.drawRect(box)
            p.setPen(theme.qcolor(theme.INK))
            p.drawText(box, Qt.AlignCenter, text)
            return
        # a thin pigment sweep along the top of the domain while the GPU compiles or the server is reached
        if self.state in ("compiling", "connecting"):
            x = r.left() + self._sweep * r.width()
            g = QLinearGradient(x - 120, 0, x + 40, 0)
            g.setColorAt(0, theme.qcolor(theme.PIGMENTS["ochre"], 0.0))
            g.setColorAt(1, theme.qcolor(theme.PIGMENTS["ochre"], 0.9))
            p.setPen(Qt.NoPen)
            p.setBrush(g)
            p.drawRect(QRectF(max(r.left(), x - 120), r.top() - 3, min(160, r.right() - max(r.left(), x - 120)), 3))
        if self.state_text:
            p.setFont(theme.font("text", 14, 500))
            fm = p.fontMetrics()
            lines = self.state_text.split("\n")
            box_w = max(fm.horizontalAdvance(s) for s in lines) + 36
            box_h = fm.height() * len(lines) + 26
            box = QRectF(r.center().x() - box_w / 2, r.center().y() - box_h / 2, box_w, box_h)
            p.setBrush(theme.qcolor(theme.PAPER, 0.94))
            p.setPen(QPen(theme.qcolor(theme.INK, 0.25), 1))
            p.drawRoundedRect(box, 6, 6)
            p.setPen(theme.qcolor(theme.INK))
            for i, s in enumerate(lines):
                p.drawText(QPointF(box.left() + 18, box.top() + 13 + fm.ascent() + i * fm.height()), s)


def _fmt(v: float) -> str:
    a = abs(v)
    if a == 0:
        return "0"
    if a < 1e-2 or a >= 1e4:
        return theme.sci(v)
    return f"{v:.3g}"
