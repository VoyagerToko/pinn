"""Talking to scripts/serve.py: finding it, starting it, and decoding its answers.

The server normally runs inside WSL on this machine. Windows' localhost forwarding into WSL adds 35-50 ms to
every request that carries a body (measured), so the viewer prefers WSL's own address.
"""
from __future__ import annotations

import json
import os
import struct
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DISTRO = os.environ.get("PINN_WSL_DISTRO", "Ubuntu")
_NO_WINDOW = 0x08000000 if os.name == "nt" else 0


def wsl_path(p: Path) -> str:
    """C:\\Users\\x\\pinn -> /mnt/c/Users/x/pinn"""
    p = Path(p).resolve()
    drive = p.drive.rstrip(":").lower()
    return f"/mnt/{drive}{p.as_posix()[len(p.drive):]}"


def wsl_address(timeout: float = 8.0) -> Optional[str]:
    try:
        out = subprocess.run(["wsl", "-d", DISTRO, "--", "hostname", "-I"], capture_output=True, timeout=timeout, creationflags=_NO_WINDOW)
    except Exception:
        return None
    text = out.stdout.replace(b"\x00", b"").decode("utf-8", errors="ignore").split()
    return text[0] if text else None


def candidate_urls() -> List[str]:
    urls = []
    if os.environ.get("PINN_SERVER"):
        urls.append(os.environ["PINN_SERVER"].rstrip("/"))
    ip = wsl_address()
    if ip:
        urls.append(f"http://{ip}:8000")
    urls.append("http://127.0.0.1:8000")
    return urls


def start_wsl_server() -> bool:
    """Launch scripts/serve_local_wsl.sh in WSL without blocking. Returns False when WSL cannot be started."""
    try:
        subprocess.Popen(
            ["wsl", "-d", DISTRO, "--", "bash", wsl_path(ROOT / "scripts" / "serve_local_wsl.sh")],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, creationflags=_NO_WINDOW,
        )
        return True
    except Exception:
        return False


def unpack(buf: bytes) -> Tuple[Dict, Dict[str, np.ndarray]]:
    """Inverse of serve.pack_fields."""
    n = struct.unpack("<I", buf[:4])[0]
    header = json.loads(buf[4 : 4 + n])
    body = memoryview(buf)[4 + n :]
    dt = np.dtype(header.get("dtype", "float32"))
    size = int(np.prod(header["shape"]))
    nb = size * dt.itemsize
    fields = {k: np.frombuffer(body[i * nb : (i + 1) * nb], dtype=dt).astype(np.float32).reshape(header["shape"]) for i, k in enumerate(header["fields"])}
    return header, fields


class Client:
    """One HTTP connection to the server. Create one per thread."""

    def __init__(self, base: str, timeout: float = 600.0):
        import httpx

        self.base = base.rstrip("/")
        self.http = httpx.Client(timeout=httpx.Timeout(timeout, connect=3.0), headers={"Accept-Encoding": "identity"})

    def close(self):
        self.http.close()

    def health(self, timeout: float = 2.0) -> Dict:
        return self.http.get(f"{self.base}/health", timeout=timeout).json()

    def models(self) -> List[Dict]:
        return self.http.get(f"{self.base}/models").json()

    def field(self, sid: str, t: float, Re: float, n: int) -> Tuple[Dict, Dict[str, np.ndarray], float]:
        r = self.http.get(f"{self.base}/field/{sid}", params={"t": t, "Re": Re, "n": n, "dtype": "float32"})
        r.raise_for_status()
        hdr, f = unpack(r.content)
        return hdr, f, float(r.headers.get("x-compute-seconds", "nan"))

    def curve(self, sid: str) -> Dict:
        r = self.http.get(f"{self.base}/curve/{sid}")
        r.raise_for_status()
        return r.json()

    def points(self, sid: str) -> Dict:
        return self.http.get(f"{self.base}/points/{sid}").json()

    def velocity(self, sid: str, pts: np.ndarray, t: float, Re: float) -> Tuple[np.ndarray, float]:
        dim = pts.shape[1]
        r = self.http.post(f"{self.base}/velocity/{sid}", params={"t": t, "Re": Re, "dim": dim}, content=np.ascontiguousarray(pts, dtype="<f4").tobytes())
        r.raise_for_status()
        return np.frombuffer(r.content, dtype="<f4").reshape(-1, dim).copy(), float(r.headers.get("x-compute-seconds", "nan"))


def find_server(urls: Optional[List[str]] = None) -> Optional[Tuple[str, Dict]]:
    """First URL whose /health answers, with that answer."""
    import httpx

    for url in urls or candidate_urls():
        try:
            h = httpx.get(f"{url}/health", timeout=1.5).json()
            if h.get("ok"):
                return url, h
        except Exception:
            continue
    return None
