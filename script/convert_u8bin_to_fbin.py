#!/usr/bin/env python3
"""Streaming converter: big-ann BIN uint8 (.u8bin) -> BIN float32 (.fbin).

The BIN layout is an 8-byte header [n, d] (two little-endian int32) followed by
n*d values, row-major. This casts each uint8 value to float32 and re-emits the
same BIN layout that ``data_loader.read_fbin`` expects.

The conversion is fully streaming: only ``CHUNK_ROWS * d`` elements are held in
memory at a time, so it runs comfortably on a low-RAM box (e.g. 14 GB) even for
1B-vector / hundreds-of-GB files.

Usage:
    python convert_u8bin_to_fbin.py [SRC] [DST] [CHUNK_ROWS]
Defaults: SRC=<repo>/data/sift1b/base.1B.u8bin
          DST=<SRC with .u8bin -> .fbin>
"""
import os
import struct
import sys
import time

import numpy as np

DEFAULT_SRC = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "sift1b",
    "base.1B.u8bin",
)


def convert(src: str, dst: str, chunk_rows: int) -> None:
    with open(src, "rb") as f:
        n, d = struct.unpack("<ii", f.read(8))
    body = os.path.getsize(src) - 8
    if n <= 0 or d <= 0 or n * d != body:
        raise SystemExit(f"header n={n} d={d} inconsistent with file body {body}")

    in_gb = body / 1e9
    out_gb = (8 + n * d * 4) / 1e9
    free_gb = shutil_disk_free(dst)
    print(
        f"converting: {src}\n"
        f"  n={n:,} d={d}  in~{in_gb:.1f}GB -> out~{out_gb:.1f}GB "
        f"(free on dest fs~{free_gb:.0f}GB)",
        flush=True,
    )
    if free_gb is not None and free_gb < out_gb * 1.05:
        raise SystemExit(f"not enough free space on destination fs ({free_gb:.0f}GB)")

    src_f = open(src, "rb")
    src_f.seek(8)
    done = 0
    t0 = time.time()
    try:
        with open(dst, "wb") as out:
            out.write(struct.pack("<ii", n, d))
            while done < n:
                rows = min(chunk_rows, n - done)
                buf = np.fromfile(src_f, dtype=np.uint8, count=rows * d)
                if buf.size != rows * d:
                    raise IOError(f"short read at row {done}: {buf.size}/{rows * d}")
                # astype -> C-contiguous float32; file.write consumes the buffer
                # directly (buffer protocol), no extra tobytes() copy.
                out.write(buf.astype(np.float32))
                done += rows
                if (done // chunk_rows) % 10 == 0 or done == n:
                    el = max(time.time() - t0, 1e-6)
                    gb_out = (done * d * 4) / 1e9
                    print(
                        f"  {done:,}/{n:,} rows ({100 * done / n:5.1f}%)  "
                        f"out {gb_out:6.1f}GB  {el:7.0f}s  {gb_out / el:5.1f}GB/s",
                        flush=True,
                    )
    finally:
        src_f.close()

    sz = os.path.getsize(dst)
    ok = sz == 8 + n * d * 4
    print(f"DONE  {dst}  size={sz / 1e9:.1f}GB  match={ok}", flush=True)
    if not ok:
        raise SystemExit("output size mismatch!")


def shutil_disk_free(path: str):
    try:
        st = os.statvfs(os.path.dirname(os.path.abspath(path)) or "/")
        return st.f_bavail * st.f_frsize / 1e9
    except OSError:
        return None


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_SRC
    dst = sys.argv[2] if len(sys.argv) > 2 else src.replace(".u8bin", ".fbin")
    if dst == src:
        dst = src + ".fbin"
    chunk_rows = int(sys.argv[3]) if len(sys.argv) > 3 else 1_000_000
    convert(src, dst, chunk_rows)


if __name__ == "__main__":
    main()
