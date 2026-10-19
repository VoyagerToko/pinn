"""The drying rack: the pulled print of the reference and the one chart that carries each scene's evidence.

The rack is a column beside a tray that hugs a square or near-square domain, and a row beneath a full-width tray
(the cylinder channel). Charts are matplotlib in the world's palette; the recovered coefficients are painted in Qt so
their figures can be tabular and aligned on the decimal point.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QFontMetricsF, QImage, QPainter, QPen
from PySide6.QtWidgets import QBoxLayout, QLabel, QSizePolicy, QStackedWidget, QVBoxLayout, QWidget

from viewer import materials, theme


class PrintSheet(QWidget):
    """A pulled sheet (procedural paper with a deckled edge and its blurred shadow); the reference is printed on it."""

    PAPER_MARGIN = 12  # logical px between the widget edge and the paper, room for the deckle and the shadow

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.image: Optional[QImage] = None
        self.domain = [0, 1, 0, 1]
        self.title = "Reference"
        self.note = ""
        self.empty = ""
        self.cylinder = None

    def _sheet(self) -> QRectF:
        m = self.PAPER_MARGIN
        return QRectF(self.rect()).adjusted(m, m, -m, -m)

    def image_rect(self) -> QRectF:
        s = self._sheet().adjusted(18, 44, -18, -38)
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
        dpr = self.devicePixelRatioF()
        W, H = int(self.width() * dpr), int(self.height() * dpr)
        if W > 40 and H > 40:
            paper, shadow = materials.paper_sheet(W, H, margin=int(self.PAPER_MARGIN * dpr))
            p.drawImage(QRectF(self.rect()), shadow)
            p.drawImage(QRectF(self.rect()), paper)
        sheet = self._sheet()
        p.setFont(theme.font("display", 19, 550))
        p.setPen(theme.qcolor(theme.INK))
        p.drawText(QPointF(sheet.left() + 18, sheet.top() + 30), self.title)
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
            p.setFont(theme.font("text", 14, 400))
            p.setPen(theme.qcolor(theme.INK_SOFT))
            p.drawText(sheet.adjusted(18, 42, -18, -14), Qt.AlignLeft | Qt.AlignTop | Qt.TextWordWrap, self.empty)
        if self.note and self.image is not None:
            p.setFont(theme.font("text", theme.SMALL, 400))
            p.setPen(theme.qcolor(theme.INK_SOFT))
            p.drawText(QRectF(sheet.left() + 18, sheet.bottom() - 30, sheet.width() - 36, 20), Qt.AlignLeft | Qt.AlignVCenter, self.note)
        p.end()


class CoefficientPanel(QWidget):
    """The inverse problem's evidence: each recovered coefficient beside its true value, decimals aligned."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.rows = []  # (name, value, truth)

    def set(self, data: Dict):
        self.rows = [("convection coefficient", data.get("lambda_1"), 1.0), ("viscosity", data.get("lambda_2"), 0.01)]
        self.update()

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        r = QRectF(self.rect()).adjusted(12, 6, -12, -6)
        p.setPen(theme.qcolor(theme.INK))
        p.setFont(theme.font("text", 15, 600))
        p.drawText(QPointF(r.left(), r.top() + 18), "Coefficients recovered from 5,000 velocity samples")
        vf = theme.font("text", 30, 550, tabular=True)
        sf = theme.font("text", 14, 400, tabular=True)
        nf = theme.font("text", 14, 500)
        fmv = QFontMetricsF(vf)
        point_x = r.left() + max(170.0, r.width() * 0.42)  # every value's decimal point sits on this x
        top = r.top() + 44
        row_h = min(86.0, (r.height() - 44) / 2)
        for i, (name, v, truth) in enumerate(self.rows):
            y = top + i * row_h
            p.setPen(Qt.NoPen)
            p.setBrush(theme.qcolor(theme.PIGMENTS["ochre"]))  # ochre marks the scene, never carries text
            p.drawEllipse(QPointF(r.left() + 4, y + 19), 4, 4)
            p.setFont(nf)
            p.setPen(theme.qcolor(theme.INK_SOFT))
            p.drawText(QPointF(r.left() + 16, y + 24), name)
            if v is None:
                continue
            s = f"{v:.6f}"
            whole, frac = s.split(".")
            p.setFont(vf)
            p.setPen(theme.qcolor(theme.INK))
            base = y + 58
            p.drawText(QPointF(point_x - fmv.horizontalAdvance(whole), base), whole)
            p.drawText(QPointF(point_x, base), "." + frac)
            err = abs(v - truth) / abs(truth) * 100
            p.setFont(sf)
            p.setPen(theme.qcolor(theme.INK_SOFT))
            x = point_x + fmv.horizontalAdvance("." + frac) + 22
            p.drawText(QPointF(x, base - 16), f"true {truth:g}")
            p.drawText(QPointF(x, base + 2), f"off by {err:.3f} %")
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
        self.canvas.setFocusPolicy(Qt.NoFocus)
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

    def _ax(self, title: str, pad: float = 9):
        ax = self.fig.add_subplot(111)
        ax.set_title(title, loc="left", color=theme.INK, fontsize=11, fontweight="semibold", pad=pad)
        ax.tick_params(length=3, width=0.6)
        for s in ("left", "bottom"):
            ax.spines[s].set_linewidth(0.7)
        return ax

    def _cursor(self, ax, t):
        # the scene's time, dashed and kept off the spine by the axis margin, so it never reads as a second axis
        return ax.axvline(t or 0.0, color=theme.INK, lw=1.0, ls=(0, (2, 2)), alpha=0.8, zorder=1)

    def draw(self, kind: str, data: Dict, pigment: str, t: Optional[float] = None, Re: Optional[float] = None, line: Optional[Dict] = None):
        import matplotlib

        self.kind = kind
        with matplotlib.rc_context(theme.mpl_style()):
            self.fig.clf()
            ink, soft = theme.INK, theme.INK_SOFT
            if kind == "error_time":
                ax = self._ax("Velocity error against the exact solution")
                ax.semilogy(data["t"], data["series"]["rel_l2_velocity"], color=pigment, lw=1.8, label="network, live grid", zorder=3)
                ax.axhline(1e-4, color=ink, lw=0.9, ls=(0, (4, 3)))
                ax.text(data["t"][-1], 1.1e-4, "gate 1e-4", ha="right", va="bottom", color=ink, fontsize=9.5)
                ax.set_xlabel("t")
                ax.margins(x=0.03)
                self.cursor = self._cursor(ax, t)
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
                    ax.plot(ghia["u"], ghia["y"], "o", ms=3.8, mfc=theme.PAPER, mec=ink, mew=0.9, label="Ghia et al. 1982")
                ax.set_xlabel("u")
                ax.set_ylabel("y")
                ax.legend(loc="lower right", fontsize=9.5, labelcolor=soft)
            elif kind == "flowrate":
                ax = self._ax("Flow rate downstream, share of the inflow")
                ts = np.asarray(data["t"])
                i = int(np.argmin(np.abs(ts - (t or 0.0))))
                st = np.asarray(data["stations"])
                ratio = np.asarray(data["flow_rate_ratio"], float)[i]
                ok = np.isfinite(ratio)  # a station without a value is left out, not drawn as zero
                ax.bar(st[ok], ratio[ok], width=0.07, color=pigment, alpha=0.78)
                ax.plot(st[ok], ratio[ok], "o", ms=4.5, color=pigment, mec=theme.INK, mew=0.6)
                ax.axhline(1.0, color=ink, lw=0.9, ls=(0, (4, 3)))
                ax.text(st[-1] + 0.12, 1.02, "mass conserved", ha="right", va="bottom", color=ink, fontsize=9.5)
                ax.set_ylim(0, 1.25)
                ax.set_xlabel(f"x [m]   (t = {ts[i]:.2f} s)")
            elif kind == "dissipation":
                ax = self._ax("Kinetic-energy dissipation", pad=30)
                ax.plot(data["ref"]["t"], data["ref"]["dissipation_dns"], color=ink, lw=1.6, label="DNS reference", zorder=3)
                ax.plot(data["t"], data["series"]["dissipation_from_energy"], color=pigment, lw=1.8, label="network, -dE/dt", zorder=3)
                ax.plot(data["t"], data["series"]["dissipation_from_enstrophy"], color=pigment, lw=1.2, ls=(0, (3, 2)), label="network, from enstrophy", zorder=3)
                g = data.get("gate", {})
                if g:
                    ax.plot([g["peak_time"]], [g["peak_value"]], "o", ms=5, mfc=theme.PAPER, mec=ink, zorder=4)
                ax.set_xlabel("t")
                ax.margins(x=0.03)
                # the key sits above the plot, where the moving time cursor never crosses it
                ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=3, fontsize=9.5, labelcolor=soft,
                          handlelength=1.8, columnspacing=1.4, borderaxespad=0.2)
                self.cursor = self._cursor(ax, t)
            self.fig.tight_layout(pad=0.6)
        self.canvas.draw_idle()

    def move_cursor(self, t: float):
        if self.cursor is not None:
            self.cursor.set_xdata([t, t])
            self.canvas.draw_idle()


class Rack(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QBoxLayout(QBoxLayout.TopToBottom, self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(12)
        self.sheet = PrintSheet()
        self.chart = EvidenceChart()
        self.coeffs = CoefficientPanel()
        self.evidence = QStackedWidget()
        self.evidence.addWidget(self.chart)
        self.evidence.addWidget(self.coeffs)
        self.caption = QLabel()
        self.caption.setWordWrap(True)
        self.caption.setFont(theme.font("text", 14.5, 400))
        self.caption.setStyleSheet(f"color: {theme.INK};")
        self.caption.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.caption.setTextInteractionFlags(Qt.NoTextInteraction)
        lay.addWidget(self.sheet, 11)
        lay.addWidget(self.evidence, 8)
        lay.addWidget(self.caption, 0)
        self._lay = lay
        self.compact = False
        self.direction = "column"

    def show_coefficients(self, on: bool):
        self.evidence.setCurrentWidget(self.coeffs if on else self.chart)

    def set_compact(self, compact: bool):
        """Scenes without a reference field: the sheet shrinks to a note and the evidence takes the room."""
        self.compact = compact
        self.arrange(self.direction)

    def arrange(self, direction: str):
        self.direction = direction
        big = 16777215
        lay, c = self._lay, self.compact
        if direction == "row":
            lay.setDirection(QBoxLayout.LeftToRight)
            lay.setSpacing(24)
            self.sheet.setMinimumHeight(150 if c else 0)
            self.sheet.setMaximumHeight(150 if c else big)
            self.sheet.setFixedWidth(400) if c else (self.sheet.setMinimumWidth(0), self.sheet.setMaximumWidth(big))
            self.caption.setFixedWidth(360)
            lay.setAlignment(self.sheet, Qt.AlignTop if c else Qt.Alignment())
            lay.setStretch(0, 0 if c else 4)
            lay.setStretch(1, 7)
            lay.setStretch(2, 0)
        else:
            lay.setDirection(QBoxLayout.TopToBottom)
            lay.setSpacing(12)
            self.sheet.setMinimumWidth(0)
            self.sheet.setMaximumWidth(big)
            self.caption.setMinimumWidth(0)
            self.caption.setMaximumWidth(big)
            lay.setAlignment(self.sheet, Qt.Alignment())
            self.sheet.setMaximumHeight(150 if c else big)
            self.sheet.setMinimumHeight(150 if c else 0)
            lay.setStretch(0, 0 if c else 11)
            lay.setStretch(1, 8)
            lay.setStretch(2, 0)
