"""The tray as an object. Its top lip carries the scene's title and summary on the left and the verdict stamp on the
right; the water holds the stage (the 2D bath or the 3D tank); the bottom lip carries one line on how the pigment is
moving. One tray for every scene, so the 3D tank sits in the same wet frame as the 2D baths.

Body lays the tray and the rack out from the domain's aspect ratio: the tray hugs the water (no dead ground beside
it) and the rack takes the rest of the width; a domain wider than about 2.2:1 gets a full-width tray with the rack
as a row beneath it.
"""
from __future__ import annotations

from PySide6.QtCore import QRect, QRectF, Qt
from PySide6.QtGui import QPainter, QPen
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

from viewer import materials, theme


class TitleBlock(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(2)
        self.title = QLabel("Connecting")
        self.title.setFont(theme.font("display", 31, 600))
        self.title.setFixedHeight(42)
        self.title.setStyleSheet(f"color: {theme.INK};")
        self.summary = QLabel("")
        self.summary.setWordWrap(True)
        self.summary.setFont(theme.font("text", 14, 400))
        self.summary.setStyleSheet(f"color: {theme.INK_SOFT};")
        self.summary.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        lay.addWidget(self.title)
        lay.addWidget(self.summary, 1)


class BathPanel(QWidget):
    OUT = 10      # room for the shadow around the tray
    PAD = 14      # lip width at the sides
    LIP = 132     # top lip: title and up to four lines of summary, stamp
    FOOT = 36     # bottom lip: the mode line
    INSET = 5     # water edge to stage

    def __init__(self, stamp, stage, parent=None):
        super().__init__(parent)
        self.head = TitleBlock(self)
        self.stamp, self.stage = stamp, stage
        stamp.setParent(self)
        stage.setParent(self)
        self.mode_line = QLabel("", self)
        self.mode_line.setFont(theme.font("text", theme.SMALL, 400))
        self.mode_line.setStyleSheet(f"color: {theme.INK_SOFT};")
        self.live_line = QLabel("", self)  # live-derived: hairline, beside the mode line on the bottom lip
        self.live_line.setFont(theme.font("text", theme.SMALL, 400, tabular=True))
        self.live_line.setStyleSheet(f"color: {theme.HAIR};")
        self.live_line.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

    def set_live(self, text: str):
        if text != self.live_line.text():
            self.live_line.setText(text)
            self._place_foot()

    def _place_foot(self):
        o, p = self.OUT, self.PAD
        wr = self.water_rect()
        live_w = self.live_line.fontMetrics().horizontalAdvance(self.live_line.text()) + 4 if self.live_line.text() else 0
        self.mode_line.setGeometry(o + p + 6, wr.bottom() + 1, wr.width() - 12 - live_w - 16, self.FOOT - 4)
        self.live_line.setGeometry(wr.right() - live_w - 6, wr.bottom() + 1, live_w, self.FOOT - 4)

    # ---- geometry, shared with Body so it can size the tray to the water ----
    @classmethod
    def chrome(cls):
        """(horizontal, vertical) pixels the tray adds around its stage."""
        return 2 * (cls.OUT + cls.PAD + cls.INSET), 2 * cls.OUT + cls.LIP + cls.FOOT + 2 * cls.INSET

    def min_width(self) -> int:
        """Narrowest tray whose lip holds the title on one line, a readable summary column and the stamp."""
        from PySide6.QtGui import QFontMetricsF

        title_w = QFontMetricsF(self.head.title.font()).horizontalAdvance(self.head.title.text())
        room = self.LIP - 18 - self.head.title.height() - self.head.layout().spacing()
        col = max(title_w + 8, 340)
        while col < 900 and self.head.summary.heightForWidth(int(col)) > room:
            col += 10
        return int(2 * (self.OUT + self.PAD) + 6 + col + 26 + self.stamp.width())

    def water_rect(self) -> QRect:
        o, p = self.OUT, self.PAD
        return QRect(o + p, o + self.LIP, self.width() - 2 * (o + p), self.height() - 2 * o - self.LIP - self.FOOT)

    def resizeEvent(self, e):
        o, p = self.OUT, self.PAD
        sw = self.stamp.width()
        self.stamp.move(self.width() - o - p - sw + 6, o + 6)
        self.head.setGeometry(o + p + 6, o + 12, max(80, self.width() - 2 * (o + p) - sw - 26), self.LIP - 18)
        wr = self.water_rect()
        i = self.INSET
        self.stage.setGeometry(wr.adjusted(i, i, -i, -i))
        self._place_foot()
        super().resizeEvent(e)

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        dpr = self.devicePixelRatioF()
        W, H = int(self.width() * dpr), int(self.height() * dpr)
        if W > 40 and H > 40:
            p.drawImage(QRectF(self.rect()), materials.soft_shadow(W, H, 16 * dpr, int(self.OUT * dpr), blur=6 * dpr, dy=int(4 * dpr)))
        o = self.OUT
        tray = QRectF(self.rect()).adjusted(o, o, -o, -o)
        p.setPen(Qt.NoPen)
        p.setBrush(theme.qcolor(theme.SIZE_DEEP))
        p.drawRoundedRect(tray, 16, 16)
        p.setBrush(theme.qcolor(theme.RIM))  # the lip, a lighter band of the tray's own material
        p.drawRoundedRect(tray.adjusted(2.5, 2.5, -2.5, -2.5), 13.5, 13.5)
        water = QRectF(self.water_rect())
        p.setBrush(theme.qcolor(theme.SIZE_DEEP))  # the lip's inner wall, then the water
        p.drawRoundedRect(water.adjusted(-3, -3, 3, 3), 11, 11)
        p.setBrush(theme.qcolor(theme.SIZE_WET))
        p.drawRoundedRect(water, 8, 8)
        p.setBrush(Qt.NoBrush)
        p.setPen(QPen(theme.qcolor(theme.PAPER, 0.6), 1))  # light catching the lip's edge
        p.drawRoundedRect(water.adjusted(-3.5, -3.5, 3.5, 3.5), 11.5, 11.5)
        p.end()


class Body(QWidget):
    """Tray and rack, laid out from the current scene's domain aspect ratio."""

    GAP = 24
    WIDE = 2.2           # domains wider than this get the full-width tray
    MAX_TRAY_FRAC = 0.70
    BATH_EXTRA = (2 * 10, 2 * 10 + 28)  # Bath2D's own margin and legend strip around the domain

    def __init__(self, panel: BathPanel, rack, parent=None):
        super().__init__(parent)
        self.panel, self.rack = panel, rack
        panel.setParent(self)
        rack.setParent(self)
        self.aspect = 1.0
        self.wide = False

    def set_aspect(self, aspect: float):
        self.aspect = float(aspect)
        self._layout()

    def resizeEvent(self, e):
        self._layout()
        super().resizeEvent(e)

    def _layout(self):
        W, H = self.width(), self.height()
        if W < 50 or H < 50:
            return
        cx, cy = BathPanel.chrome()
        bx, by = self.BATH_EXTRA
        self.wide = self.aspect > self.WIDE
        if self.wide:
            dom_w = W - cx - bx
            dom_h = dom_w / self.aspect
            ph = int(min(dom_h + by + cy, H * 0.68))
            self.panel.setGeometry(0, 0, W, ph)
            self.rack.arrange("row")
            self.rack.setGeometry(0, ph + self.GAP // 2, W, H - ph - self.GAP // 2)
        else:
            dom_h = H - cy - by
            pw = int(dom_h * self.aspect + bx + cx)
            pw = max(min(pw, int(W * self.MAX_TRAY_FRAC)), min(self.panel.min_width(), int(W * self.MAX_TRAY_FRAC)))
            self.panel.setGeometry(0, 0, pw, H)
            self.rack.arrange("column")
            self.rack.setGeometry(pw + self.GAP, 0, W - pw - self.GAP, H)
