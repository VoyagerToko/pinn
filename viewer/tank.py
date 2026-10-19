"""The glass tank for the 3D scene: pigment carried through the box, vortex tubes from the network's Q-criterion."""
from __future__ import annotations

from typing import Optional

import numpy as np
from PySide6.QtWidgets import QVBoxLayout, QWidget

from viewer import theme


class Tank3D(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        import pyvista as pv
        from pyvistaqt import QtInteractor

        self.pv = pv
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)  # the tray (panel.BathPanel) frames the tank
        self.plotter = QtInteractor(self, auto_update=False)
        lay.addWidget(self.plotter.interactor)
        pl = self.plotter
        pl.set_background(theme.SIZE_WET)
        pl.enable_anti_aliasing("ssaa") if hasattr(pl, "enable_anti_aliasing") else None
        L = 2 * np.pi
        box = pv.Box(bounds=(0, L, 0, L, 0, L))
        pl.add_mesh(box.extract_feature_edges(), color=theme.INK, line_width=1.2, opacity=0.55)
        self.L = L
        self.points = None
        self.points_actor = None
        self.tubes_actor = None
        pl.camera_position = [(L * 2.1, -L * 1.5, L * 1.6), (L / 2, L / 2, L / 2), (0, 0, 1)]
        pl.camera.zoom(1.05)
        self.orbit = True

    def update_particles(self, hist: np.ndarray, active: np.ndarray, pig: np.ndarray, colors: np.ndarray, playing: bool):
        K, N, _ = hist.shape
        pts = hist.reshape(-1, 3).astype(np.float32)
        rgb = colors[np.tile(pig, K)].astype(np.uint8)
        age = np.repeat(np.arange(K), N)
        alpha = np.where(np.tile(active, K), (255 * (1.0 - age / K) ** 1.4), 0).astype(np.uint8)
        rgba = np.concatenate([rgb, alpha[:, None]], 1)
        if self.points is None or self.points.n_points != len(pts):
            self.points = self.pv.PolyData(pts)
            self.points.point_data["rgba"] = rgba
            self.points_actor = self.plotter.add_mesh(self.points, scalars="rgba", rgba=True, render_points_as_spheres=True, point_size=5.5, show_scalar_bar=False, reset_camera=False)
        else:
            self.points.points = pts
            self.points.point_data["rgba"] = rgba
        if playing and self.orbit:
            self.plotter.camera.azimuth = self.plotter.camera.azimuth + 0.12
        self.plotter.render()

    def update_field(self, fields, n: int, level_frac: float = 0.1):
        pv = self.pv
        q = fields.get("q")
        if q is None:
            return
        grid = pv.ImageData(dimensions=(n, n, n), spacing=(self.L / n,) * 3)
        grid.point_data["q"] = np.asarray(q, np.float32).ravel(order="F")
        qmax = float(np.nanmax(q))
        if qmax <= 0:
            return
        tubes = grid.contour(isosurfaces=[level_frac * qmax], scalars="q")
        if self.tubes_actor is not None:
            self.plotter.remove_actor(self.tubes_actor, render=False)
            self.tubes_actor = None
        if tubes.n_points:
            self.tubes_actor = self.plotter.add_mesh(tubes, color=theme.PIGMENTS["cobalt violet"], opacity=0.28, smooth_shading=True, specular=0.2, show_scalar_bar=False, reset_camera=False)
        self.plotter.render()

    def grab_image(self):
        return self.plotter.screenshot(return_img=True)

    def close_plotter(self):
        try:
            self.plotter.close()
        except Exception:
            pass
