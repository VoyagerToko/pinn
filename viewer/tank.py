"""The glass tank for 3D scenes.

box     the periodic Taylor-Green cube: pigment carried through it, vortex tubes from the network's Q-criterion
vessel  a real aorta from a 4D flow MRI scan: the lumen as a translucent surface built from the scan's own vessel mask,
        its voxels coloured by the chosen view (scan speed, network speed or pressure), pigment released at the upstream
        catheter plane and carried by the network
"""
from __future__ import annotations

from typing import Dict, Optional

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
        self.mode = None
        self.points = None
        self.points_actor = None
        self.tubes_actor = None
        self.frame_actors = []
        self.cloud = None
        self.cloud_actor = None
        self.orbit = True
        self.L = 2 * np.pi
        self.configure_box()

    # ---- modes ---------------------------------------------------------------------------------------
    def _clear(self):
        for a in self.frame_actors + [self.points_actor, self.tubes_actor, self.cloud_actor]:
            if a is not None:
                self.plotter.remove_actor(a, render=False)
        self.frame_actors, self.points_actor, self.tubes_actor, self.cloud_actor = [], None, None, None
        self.points, self.cloud = None, None

    def configure_box(self):
        if self.mode == "box":
            return
        self._clear()
        pv, pl, L = self.pv, self.plotter, self.L
        box = pv.Box(bounds=(0, L, 0, L, 0, L))
        self.frame_actors.append(pl.add_mesh(box.extract_feature_edges(), color=theme.INK, line_width=1.2, opacity=0.55))
        pl.camera_position = [(L * 2.1, -L * 1.5, L * 1.6), (L / 2, L / 2, L / 2), (0, 0, 1)]
        pl.camera.zoom(1.05)
        self.orbit = True
        self.mode = "box"
        pl.render()

    def configure_vessel(self, geom: Dict):
        """geom: the /points payload of the coarctation scene (lumen voxel positions and grid indices, voxel size)."""
        from scipy import ndimage

        self._clear()
        pv, pl = self.pv, self.plotter
        ijk = np.asarray(geom["ijk"], int)
        voxel = np.asarray(geom["voxel"], float)
        origin = np.asarray(geom["origin"], float)
        dims = ijk.max(0) + 3
        mask = np.zeros(dims, np.float32)
        mask[tuple((ijk + 1).T)] = 1.0
        mask = ndimage.gaussian_filter(mask, 0.8)  # a smooth wall rather than voxel steps
        grid = pv.ImageData(dimensions=tuple(int(d) for d in dims), spacing=tuple(voxel), origin=tuple(origin - voxel))
        grid.point_data["m"] = mask.ravel(order="F")
        wall = grid.contour(isosurfaces=[0.5], scalars="m")
        self.frame_actors.append(pl.add_mesh(wall, color=theme.PAPER, opacity=0.16, smooth_shading=True, specular=0.15, show_scalar_bar=False))
        pos = np.asarray(geom["pos"], np.float32)
        self.cloud = pv.PolyData(pos)
        self.cloud.point_data["rgba"] = np.zeros((self.cloud.n_points, 4), np.uint8)
        self.cloud_actor = pl.add_mesh(self.cloud, scalars="rgba", rgba=True, render_points_as_spheres=True, point_size=3.0, show_scalar_bar=False)
        # the scan's frame: a = image columns, b = image rows (the head at small b), c = slices; look along c, head up
        c = 0.5 * (pos.min(0) + pos.max(0))
        span = float(np.ptp(pos, axis=0).max())
        # look along +c (from the -c side) with the head up, then fit the whole vessel in view
        pl.view_vector((0.0, 0.0, -1.0), viewup=(0.0, -1.0, 0.0))
        pl.reset_camera()
        self.orbit = False
        self.mode = "vessel"
        pl.render()

    # ---- content -------------------------------------------------------------------------------------
    def update_cloud(self, values: Optional[np.ndarray], lut: Optional[np.ndarray] = None, lo: float = 0.0, hi: float = 1.0, alpha: float = 0.85):
        """Colour the lumen voxels (vessel mode); values None hides them."""
        if self.cloud is None:
            return
        if values is None or len(values) != self.cloud.n_points:
            self.cloud.point_data["rgba"] = np.zeros((self.cloud.n_points, 4), np.uint8)
        else:
            x = np.clip((np.nan_to_num(np.asarray(values, np.float32)) - lo) / max(hi - lo, 1e-12), 0, 1)
            rgb = lut[(x * (len(lut) - 1)).astype(int)]
            a = np.full((len(x), 1), int(255 * alpha), np.uint8)
            self.cloud.point_data["rgba"] = np.concatenate([rgb, a], 1)
        self.plotter.render()

    def update_particles(self, hist: np.ndarray, active: np.ndarray, pig: np.ndarray, colors: np.ndarray, playing: bool):
        K, N, _ = hist.shape
        pts = hist.reshape(-1, 3).astype(np.float32)
        rgb = colors[np.tile(pig, K)].astype(np.uint8)
        age = np.repeat(np.arange(K), N)
        alpha = np.where(np.tile(active, K), (255 * (1.0 - age / K) ** 1.4), 0).astype(np.uint8)
        rgba = np.concatenate([rgb, alpha[:, None]], 1)
        size = 5.5 if self.mode == "box" else 4.5
        if self.points is None or self.points.n_points != len(pts):
            if self.points_actor is not None:
                self.plotter.remove_actor(self.points_actor, render=False)
            self.points = self.pv.PolyData(pts)
            self.points.point_data["rgba"] = rgba
            self.points_actor = self.plotter.add_mesh(self.points, scalars="rgba", rgba=True, render_points_as_spheres=True, point_size=size, show_scalar_bar=False, reset_camera=False)
        else:
            self.points.points = pts
            self.points.point_data["rgba"] = rgba
        if playing and self.orbit:
            self.plotter.camera.azimuth = self.plotter.camera.azimuth + 0.12
        self.plotter.render()

    def update_field(self, fields, n: int, level_frac: float = 0.1):
        if self.mode != "box":
            return
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
