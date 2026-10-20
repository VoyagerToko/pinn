"""Read a large remote zip over HTTP range requests and extract only chosen members, in parallel.

    python scripts/remote_zip.py URL --list                         # the archive's index (names, sizes)
    python scripts/remote_zip.py URL --get 'Imaging/P1_severe/4DFlow_highRes/*' --out data/stanford_coa --workers 8

Used for the public 4D-FlowP coarctation phantom (Stanford Digital Repository, 18 GB Imaging.zip): one case's
4D-flow series is a small fraction of the archive. Members already extracted with the right size are skipped.
"""
from __future__ import annotations

import argparse
import fnmatch
import io
import sys
import threading
import time
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


class HTTPRangeFile(io.RawIOBase):
    """A seekable, read-only file over HTTP range requests, with a small read-ahead buffer."""

    def __init__(self, url: str, block: int = 1 << 20):
        self.url, self.block, self.pos = url, block, 0
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=60) as r:
            self.size = int(r.headers["Content-Length"])
        self._buf_start, self._buf = -1, b""

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else (self.pos + off if whence == 1 else self.size + off)
        return self.pos

    def _fetch(self, start, end):
        for attempt in range(8):
            try:
                req = urllib.request.Request(self.url, headers={"Range": f"bytes={start}-{end}"})
                with urllib.request.urlopen(req, timeout=120) as r:
                    return r.read()
            except Exception:
                time.sleep(2 * (attempt + 1))
        raise IOError(f"range {start}-{end} failed")

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        n = min(n, self.size - self.pos)
        if n <= 0:
            return b""
        b0 = self._buf_start
        if b0 >= 0 and b0 <= self.pos and self.pos + n <= b0 + len(self._buf):
            out = self._buf[self.pos - b0 : self.pos - b0 + n]
        else:
            want = max(n, self.block)
            end = min(self.size, self.pos + want) - 1
            data = self._fetch(self.pos, end)
            self._buf_start, self._buf = self.pos, data
            out = data[:n]
        self.pos += len(out)
        return out

    def readinto(self, b):
        data = self.read(len(b))
        b[: len(data)] = data
        return len(data)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("url")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--get", nargs="*", default=[], help="glob patterns of members to extract")
    ap.add_argument("--out", default=".")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    zf = zipfile.ZipFile(HTTPRangeFile(args.url))
    infos = zf.infolist()
    if args.list:
        for zi in infos:
            print(f"{zi.file_size:>14d}  {zi.compress_size:>14d}  {zi.filename}")
        print(f"{len(infos)} members, {sum(z.file_size for z in infos) / 1e9:.2f} GB uncompressed")
        return
    todo = [zi for zi in infos if not zi.is_dir() and any(fnmatch.fnmatch(zi.filename, p) for p in args.get)]
    out = Path(args.out)
    todo = [zi for zi in todo if not ((out / zi.filename).exists() and (out / zi.filename).stat().st_size == zi.file_size)]
    total = sum(zi.compress_size for zi in todo)
    print(f"[remote_zip] {len(todo)} members, {total / 1e6:.0f} MB to fetch", flush=True)
    lock = threading.Lock()
    done = [0]
    t0 = time.time()

    import struct
    import zlib

    rf = HTTPRangeFile.__new__(HTTPRangeFile)  # a bare range fetcher (the index was read once, above)
    rf.url = args.url

    def one(zi):
        # local header: 30 fixed bytes, then the name and extra field, then the compressed data
        hdr = rf._fetch(zi.header_offset, zi.header_offset + 29)
        n_name, n_extra = struct.unpack("<HH", hdr[26:30])
        start = zi.header_offset + 30 + n_name + n_extra
        raw = rf._fetch(start, start + zi.compress_size - 1) if zi.compress_size else b""
        if zi.compress_type == zipfile.ZIP_DEFLATED:
            data = zlib.decompressobj(-15).decompress(raw)
        elif zi.compress_type == zipfile.ZIP_STORED:
            data = raw
        else:
            raise ValueError(f"unsupported compression {zi.compress_type} for {zi.filename}")
        if len(data) != zi.file_size or zlib.crc32(data) != zi.CRC:
            raise IOError(f"corrupt member {zi.filename}")
        dst = out / zi.filename
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_suffix(dst.suffix + ".part")
        tmp.write_bytes(data)
        tmp.replace(dst)
        with lock:
            done[0] += zi.compress_size
            el = time.time() - t0
            print(f"[remote_zip] {done[0] / 1e6:8.0f} / {total / 1e6:.0f} MB  {done[0] / 1e6 / max(el, 1e-3):5.1f} MB/s  {zi.filename}", flush=True)

    with ThreadPoolExecutor(args.workers) as ex:
        list(ex.map(one, todo))
    print("[remote_zip] done", flush=True)


if __name__ == "__main__":
    sys.exit(main())
