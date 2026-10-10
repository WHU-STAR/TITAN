#!/usr/bin/env python3
"""Expand a small fvecs query set + matched ivecs ground truth by concatenating
N differently-shuffled copies of the original rows (default: 1000 -> 10000).

The same per-copy permutation is applied to query and GT rows, so row i of the
new query file and row i of the new GT file still describe the same original
query and its ground truth. Originals are kept as <path>.bak (refuses to run
if the backup already exists).

Usage:
    python3 expand_query_gt.py --queries q.fvecs --groundtruth gt.ivecs \
        [--copies 10] [--seed 12345]
"""
import argparse
import os
import random
import struct
import sys


def read_rows(path):
    """Read an fvecs/ivecs file as a list of raw row bytes (header + payload)."""
    with open(path, "rb") as f:
        data = f.read()
    rows = []
    off = 0
    total = len(data)
    while off < total:
        if off + 4 > total:
            sys.exit(f"{path}: truncated row header at offset {off}")
        (k,) = struct.unpack_from("<i", data, off)
        if k <= 0:
            sys.exit(f"{path}: bad row header {k} at offset {off}")
        size = 4 + k * 4
        if off + size > total:
            sys.exit(f"{path}: truncated row at offset {off}")
        rows.append(data[off:off + size])
        off += size
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", required=True)
    ap.add_argument("--groundtruth", required=True)
    ap.add_argument("--copies", type=int, default=10)
    ap.add_argument("--seed", type=int, default=12345)
    args = ap.parse_args()

    q = read_rows(args.queries)
    g = read_rows(args.groundtruth)
    if len(q) != len(g):
        sys.exit(f"row count mismatch: {len(q)} queries vs {len(g)} gt rows")
    print(f"{args.queries}: {len(q)} rows x {len(q[0])}B")
    print(f"{args.groundtruth}: {len(g)} rows x {len(g[0])}B")

    order = []
    for c in range(args.copies):
        perm = list(range(len(q)))
        random.Random(args.seed + c).shuffle(perm)
        order.extend(perm)

    for path, rows in ((args.queries, q), (args.groundtruth, g)):
        bak = path + ".bak"
        if os.path.exists(bak):
            sys.exit(f"refusing to overwrite existing backup {bak}")
        tmp = path + ".tmp"
        with open(tmp, "wb") as f:
            for i in order:
                f.write(rows[i])
        os.replace(path, bak)
        os.replace(tmp, path)
        print(f"{path}: wrote {len(order)} rows, original kept at {bak}")


if __name__ == "__main__":
    main()
