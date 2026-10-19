"""The drying rack: the pulled print of the reference and the one chart that carries each scene's evidence."""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QImage, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QLabel, QSizePolicy, QVBoxLayout, QWidget

from viewer import theme


class PrintSheet(QWidget):
    """A pulled sheet with a deckled edge; the reference field is printed on it at the bath's colour scale."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.image: Optional[QImage] = None
        self.domain = [0, 1, 0, 1]
        self.title = "Reference"
        self.note = ""
        self.empty = ""
        self.cylinder = None
        rng = np.random.default_rng(11)
        self._deckle = rng.normal(0, 1, 160)

    def _sheet(self) -> QRectF:
        return QRectF(self.rect()).adjusted(4, 2, -8, -10)

    def _deckle_path(self, r: QRectF) -> QPainterPath:
        """Irregular paper edge: a few pixels of fibrous wander along each side."""
        d = self._deckle
        path = QPainterPath()
        n = 40
        pts = []
        for i in range(n):
            pts.append(QPointF(r.left() + r.width() * i / n, r.top() + 1.4 * d[i]))
        for i in range(n):
            pts.append(QPointF(r.right() + 1.4 * d[40 + i], r.top() + r.height() * i / n))
        for i in range(n):
            pts.append(QPointF(r.right() - r.width() * i / n, r.bottom() + 1.4 * d[80 + i]))
        for i in range(n):
            pts.append(QPointF(r.left() + 1.4 * d[120 + i], r.bottom() - r.height() * i / n))
        path.moveTo(pts[0])
        for q in pts[1:]:
            path.lineTo(q)
        path.closeSubpath()
        return path

    def image_rect(self) -> QRectF:
        s = self._sheet().adjusted(16, 40, -16, -34)
        x0, x1, y0, y1 = self.domain[:4]
        a = (x1 - x0) / max(y1 - y0, 1e-9)
        if s.width() / max(s.height(), 1) > a:
            w = s.height() * a
            return QRectF(s.left() + (s.width() - w) / 2, s.top(), w, s.height())
        h = s.width() / a
        return QRectF(s.left(), s.top() + (s.height() - h) / 2, s.width(), h)

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setRenderHint(QPainter.SmoothPixmapTransform)
        sheet = self._sheet()
        path = self._deckle_path(sheet)
        for i, a in enumerate((0.09, 0.05)):
            p.setPen(Qt.NoPen)
            p.setBrush(theme.qcolor(theme.INK, a))
            p.drawPath(path.translated(1 + i, 2.5 + 1.5 * i))
        p.setBrush(theme.qcolor(theme.PAPER))
        p.drawPath(path)
        p.setFont(theme.font("display", 17, 500))
        p.setPen(theme.qcolor(theme.INK))
        p.drawText(QPointF(sheet.left() + 16, sheet.top() + 27), self.title)
        r = self.image_rect()
        if self.image is not None:
            p.drawImage(r, self.image)
            p.setPen(QPen(theme.qcolor(theme.INK, 0.35), 1.0))
            p.setBrush(Qt.NoBrush)
            p.drawRect(r)
            if self.cylinder is not None:
                x0, x1, y0, y1 = self.domain[:4]
                cx = r.left() + (self.cylinder["center"][0] - x0) / (x1 - x0) * r.width()
                cy = r.top() + (y1 - self.cylinder["center"][1]) / (y1 - y0) * r.height()
                rr = self.cylinder["radius"] / (x1 - x0) * r.width()
                p.setBrush(theme.qcolor(theme.STONE))
                p.drawEllipse(QPointF(cx, cy), rr, rr)
        elif self.empty:
            p.setFont(theme.font("text", 13, 400))
            p.setPen(theme.qcolor(theme.INK_SOFT))
            p.drawText(sheet.adjusted(16, 38, -16, -12), Qt.AlignLeft | Qt.AlignVCenter | Qt.TextWordWrap, self.empty)
        if self.note:
            p.setFont(theme.font("text", 11.5, 400))
            p.setPen(theme.qcolor(theme.HAIR))
            p.drawText(QRectF(sheet.left() + 16, sheet.bottom() - 28, sheet.width() - 32, 20), Qt.AlignLeft | Qt.AlignVCenter, self.note)
        p.end()


class EvidenceChart(QWidget):
    """One matplotlib chart in the world's palette; the time cursor moves without a full redraw."""

    def __init__(self, parent=None):
        super().__init__(parent)
        import matplotlib

        matplotlib.use("QtAgg")
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        from matplotlib.figure import Figure

        with matplotlib.rc_context(theme.mpl_style()):
            self.fig = Figure(figsize=(4, 2.6), dpi=100)
            self.fig.patch.set_facecolor(theme.SIZE)
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.canvas.setStyleSheet(f"background: {theme.SIZE};")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.canvas)
        self.cursor = None
        self.kind = None

    def clear(self, message: str = ""):
        import matplotlib

        with matplotlib.rc_context(theme.mpl_style()):
            self.fig.clf()
            if message:
                ax = self.fig.add_subplot(111)
                ax.axis("off")
                ax.text(0.0, 0.5, message, color=theme.INK_SOFT, fontsize=10, va="center", wrap=True)
        self.cursor = None
        self.canvas.draw_idle()

    def _ax(self, title: str):
        ax = self.fig.add_subplot(111)
        ax.set_title(title, loc="left", color=theme.INK, fontsize=10.5, fontweight="semibold", pad=8)
        ax.tick_params(length=3, width=0.6)
        for s in ("left", "bottom"):
            ax.spines[s].set_linewidth(0.7)
        return ax

    def draw(self, kind: str, data: Dict, pigment: str, t: Optional[float] = None, Re: Optional[float] = None, line: Optional[Dict] = None):
        import matplotlib

        self.kind = kind
        with matplotlib.rc_context(theme.mpl_style()):
            self.fig.clf()
            ink, soft = theme.INK, theme.INK_SOFT
            if kind == "error_time":
                ax = self._ax("Velocity error against the exact solution")
                ax.semilogy(data["t"], data["series"]["rel_l2_velocity"], color=pigment, lw=1.8, label="network, live grid")
                ax.axhline(1e-4, color=ink, lw=0.9, ls=(0, (4, 3)))
                ax.text(data["t"][-1], 1.1e-4, "gate 1e-4", ha="right", va="bottom", color=ink, fontsize=9)
                ax.set_xlabel("t")
                self.cursor = ax.axvline(t or 0.0, color=ink, lw=0.8, alpha=0.6)
            elif kind in ("centreline", "centreline_live"):
                key = str(int(round(Re))) if Re is not None else None
                ax = self._ax(f"u along x = 0.5, Re {Re:.0f}" if Re is not None else "u along x = 0.5")
                if line is not None:
                    ax.plot(line["u"], line["s"], color=pigment, lw=1.9, label="network")
                elif key and key in data.get("Re", {}):
                    e = data["Re"][key]
                    ax.plot(e["u_centre"], e["s"], color=pigment, lw=1.9, label="network")
                ghia = data.get("Re", {}).get(key, {}).get("ghia") if key else None
                if ghia and (Re is None or abs(Re - float(key)) < 1):
                    ax.plot(ghia["u"], ghia["y"], "o", ms=3.6, mfc=theme.PAPER, mec=ink, mew=0.9, label="Ghia et al. 1982")
                ax.set_xlabel("u")
                ax.set_ylabel("y")
                ax.legend(loc="lower right", fontsize=8.5, labelcolor=soft)
            elif kind == "lambdas":
                ax = self.fig.add_subplot(111)
                ax.axis("off")
                ax.text(0, 0.92, "Coefficients recovered from 5,000 samples", color=ink, fontsize=10.5, fontweight="semibold", va="top")
                rows = [("convection coefficient", data.get("lambda_1"), 1.0), ("viscosity", data.get("lambda_2"), 0.01)]
                for i, (name, v, truth) in enumerate(rows):
                    y = 0.62 - i * 0.3
                    ax.text(0.0, y, name, color=soft, fontsize=10, va="center")
                    ax.text(0.48, y, f"{v:.6f}" if v is not None else "-", color=pigment, fontsize=15, va="center", family=theme._FAMILIES.get("text"))
                    ax.text(0.80, y, f"true {truth:g}", color=soft, fontsize=9.5, va="center")
            elif kind == "flowrate":
                ax = self._ax("Flow rate downstream, share of the inflow")
                ts = np.asarray(data["t"])
                i = int(np.argmin(np.abs(ts - (t or 0.0))))
                st = np.asarray(data["stations"])
                ratio = np.asarray(data["flow_rate_ratio"], float)[i]
                ax.bar(st, np.nan_to_num(ratio), width=0.22, color=pigment, alpha=0.9)
                ax.axhline(1.0, color=ink, lw=0.9, ls=(0, (4, 3)))
                ax.text(st[-1] + 0.12, 1.0, "mass conserved", ha="right", va="bottom", color=ink, fontsize=9)
                ax.set_ylim(0, 1.25)
                ax.set_xlabel(f"x [m]   (t = {ts[i]:.2f} s)")
            elif kind == "dissipation":
                ax = self._ax("Kinetic-energy dissipation")
                ax.plot(data["ref"]["t"], data["ref"]["dissipation_dns"], color=ink, lw=1.6, label="DNS reference")
                ax.plot(data["t"], data["series"]["dissipation_from_energy"], color=pigment, lw=1.8, label="network, -dE/dt")
                ax.plot(data["t"], data["series"]["dissipation_from_enstrophy"], color=pigment, lw=1.2, ls=(0, (3, 2)), label="network, from enstrophy")
                g = data.get("gate", {})
                if g:
                    ax.plot([g["peak_time"]], [g["peak_value"]], "o", ms=5, mfc=theme.PAPER, mec=ink)
                ax.set_xlabel("t")
                ax.legend(loc="upper right", fontsize=8.5, labelcolor=soft)
                self.cursor = ax.axvline(t or 0.0, color=ink, lw=0.8, alpha=0.6)
            self.fig.tight_layout(pad=0.6)
        self.canvas.draw_idle()

    def move_cursor(self, t: float):
        if self.cursor is not None:
            self.cursor.set_xdata([t, t])
            self.canvas.draw_idle()


class Rack(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(10)
        self.sheet = PrintSheet()
        self.chart = EvidenceChart()
        self.caption = QLabel()
        self.caption.setWordWrap(True)
        self.caption.setFont(theme.font("text", 13.5, 400))
        self.caption.setStyleSheet(f"color: {theme.INK};")
        self.caption.setTextInteractionFlags(Qt.NoTextInteraction)
        lay.addWidget(self.sheet, 11)
        lay.addWidget(self.chart, 8)
        lay.addWidget(self.caption, 0)
        self._lay = lay

    def set_compact(self, compact: bool):
        """Scenes without a reference field: the sheet shrinks to a note and the evidence chart takes the room."""
        self.sheet.setMaximumHeight(124 if compact else 16777215)
        self.sheet.setMinimumHeight(124 if compact else 0)
        self._lay.setStretch(0, 0 if compact else 11)
        self._lay.setStretch(1, 8)
