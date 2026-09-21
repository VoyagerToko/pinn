"""STEP 9 - the 3D visualisation and animation pipeline.

The trained PINN is a continuous function of (x, y, z, t): sample it at any resolution and
frame rate with no interpolation. PyVista/VTK imports are lazy so the training environment
does not need them.

    9.1 sample_field_to_vti     9.2 vorticity / Q-criterion isosurfaces (by AD, not finite differences)
    9.3 advect_particles         9.4 render_frame + encode_video (ffmpeg)
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Callable, Optional, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from .physics import q_criterion, velocity_gradient, vorticity_from_grad
from .utils import chunked_vmap, meshgrid_points

Array = jnp.ndarray


def sample_grid(n: int = 128, L: float = 2 * np.pi) -> tuple[np.ndarray, np.ndarray]:
    g = np.linspace(0.0, L, n, endpoint=False)
    return g, np.stack(np.meshgrid(g, g, g, indexing="ij"), axis=-1).reshape(-1, 3)


def sample_field(vel_p_fn: Callable, t: float, pts: np.ndarray, chunk: int = 65536, with_gradients: bool = True):
    """Evaluate (u, |u|, omega, Q) at time ``t`` on scattered 3D points via AD.

    ``vel_p_fn(z) -> (u, v, w, p)`` for z = (t, x, y, z). Returns a dict of numpy arrays.
    """
    z = jnp.concatenate([jnp.full((pts.shape[0], 1), t), jnp.asarray(pts)], axis=1)
    if with_gradients:
        g = velocity_gradient(vel_p_fn, dim=3, unsteady=True)

        def one(zz):
            u, grad_u = g(zz)
            return u, vorticity_from_grad(grad_u), q_criterion(grad_u)

        u, omega, q = chunked_vmap(one, z, chunk=chunk)
        return {"velocity": np.asarray(u), "speed": np.linalg.norm(np.asarray(u), axis=1), "vorticity": np.asarray(omega), "qcriterion": np.asarray(q)}
    u = chunked_vmap(lambda zz: vel_p_fn(zz)[:3], z, chunk=chunk)
    return {"velocity": np.asarray(u), "speed": np.linalg.norm(np.asarray(u), axis=1)}


def fields_to_vti(fields: dict, n: int, spacing: float, path: os.PathLike):
    """9.1 write point data on an n^3 ImageData grid (.vti) for ParaView / PyVista."""
    import pyvista as pv

    grid = pv.ImageData(dimensions=(n, n, n), spacing=(spacing,) * 3)
    for k, v in fields.items():
        grid[k] = v
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    grid.save(str(path))
    return grid


def q_isosurface(grid, level_frac: float = 0.1):
    """9.2 vortex tubes: isosurface at Q = level_frac * Q_max (Q computed by AD upstream)."""
    q = np.asarray(grid["qcriterion"])
    return grid.contour(isosurfaces=[level_frac * float(q.max())], scalars="qcriterion")


def export_frames(vel_p_fn: Callable, out_dir: os.PathLike, n: int = 128, times: Sequence[float] = np.linspace(0, 20, 600), L: float = 2 * np.pi, with_q: bool = True):
    """Sample the field at every frame time and write frame_XXXX.vti (+ q_XXXX.vtp isosurfaces)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    g, pts = sample_grid(n, L)
    spacing = g[1] - g[0]
    for frame, t in enumerate(times):
        f = sample_field(vel_p_fn, float(t), pts, with_gradients=with_q)
        grid = fields_to_vti(f, n, spacing, out_dir / f"frame_{frame:04d}.vti")
        if with_q:
            q_isosurface(grid).save(str(out_dir / f"q_{frame:04d}.vtp"))


def advect_particles(vel_fn_batched: Callable, seeds: np.ndarray, t_span=(0.0, 20.0), n_frames: int = 600, rtol: float = 1e-6):
    """9.3 integrate dx/dt = u(x, t) for many particles with an adaptive RK45 - no grid interpolation.

    ``vel_fn_batched(t, X) -> (N, 3)`` velocity at points X (N, 3) and time t.
    Returns trajectories of shape (n_frames, N, 3).
    """
    from scipy.integrate import solve_ivp

    f = jax.jit(vel_fn_batched)

    def rhs(t, y):
        return np.asarray(f(t, jnp.asarray(y.reshape(-1, 3)))).ravel()

    sol = solve_ivp(rhs, t_span, np.asarray(seeds).ravel(), method="RK45", t_eval=np.linspace(*t_span, n_frames), rtol=rtol)
    return sol.y.T.reshape(n_frames, -1, 3)


def render_frame(grid, tubes, path: os.PathLike, frame: int = 0, window_size=(1920, 1080), cmap: str = "plasma"):
    """9.4 PyVista off-screen volume render with vortex tubes and a slow orbit."""
    import pyvista as pv

    pl = pv.Plotter(off_screen=True, window_size=window_size)
    pl.add_volume(grid, scalars="speed", cmap=cmap, opacity="sigmoid")
    if tubes is not None and tubes.n_points > 0:
        pl.add_mesh(tubes, color="cyan", opacity=0.6)
    pl.camera.azimuth = frame * 0.5
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    pl.screenshot(str(path))
    pl.close()


def encode_video(frame_dir: os.PathLike, out: os.PathLike, fps: int = 60, pattern: str = "%04d.png", crf: int = 18):
    """ffmpeg -framerate fps -i pattern -c:v libx264 -pix_fmt yuv420p -crf 18 out.mp4 (uses imageio-ffmpeg's binary if ffmpeg is not on PATH)."""
    try:
        import imageio_ffmpeg

        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        ffmpeg = "ffmpeg"
    cmd = [ffmpeg, "-y", "-framerate", str(fps), "-i", str(Path(frame_dir) / pattern), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", str(crf), str(out)]
    subprocess.run(cmd, check=True)
