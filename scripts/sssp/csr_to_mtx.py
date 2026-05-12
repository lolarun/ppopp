#!/usr/bin/env python3
"""Convert binary .csr (harness format) -> MatrixMarket .mtx (real general).

CSR format (see src/io/csr_io.cpp):
    8B magic 0x4353520000000001
    4B precision tag (FP32=0x46503332, FP64=0x46503634)
    8B nv, 8B ne
    (nv+1) * 8B row_offsets (uint64)
    ne * 4B col_indices (uint32)
    ne * sizeof(W) weights (4 for FP32, 8 for FP64)
    8B CRC64 trailer (Jones polynomial; not verified here)

Usage: python3 csr_to_mtx.py <in.csr> <out.mtx>
"""
import struct
import sys
from pathlib import Path

MAGIC = 0x4353520000000001
TAG_FP32 = 0x46503332
TAG_FP64 = 0x46503634


def convert(csr_path: Path, mtx_path: Path):
    with open(csr_path, "rb") as f:
        magic = struct.unpack("<Q", f.read(8))[0]
        if magic != MAGIC:
            raise SystemExit(f"bad magic 0x{magic:016x}")
        tag = struct.unpack("<I", f.read(4))[0]
        if tag == TAG_FP32:
            wfmt, wsz = "<f", 4
        elif tag == TAG_FP64:
            wfmt, wsz = "<d", 8
        else:
            raise SystemExit(f"bad precision tag 0x{tag:08x}")

        nv, ne = struct.unpack("<QQ", f.read(16))
        print(f"  nv={nv} ne={ne} prec={'FP32' if wsz==4 else 'FP64'}")

        row_offsets = struct.unpack(f"<{nv+1}Q", f.read((nv + 1) * 8))
        col = struct.unpack(f"<{ne}I", f.read(ne * 4))
        weights = struct.unpack(f"<{ne}{wfmt[1]}", f.read(ne * wsz))

    with open(mtx_path, "w") as out:
        out.write("%%MatrixMarket matrix coordinate real general\n")
        out.write(f"% Converted from {csr_path.name}\n")
        out.write(f"{nv} {nv} {ne}\n")
        for u in range(nv):
            for k in range(row_offsets[u], row_offsets[u + 1]):
                v = col[k]
                w = weights[k]
                out.write(f"{u + 1} {v + 1} {w}\n")
    sz = mtx_path.stat().st_size / 1024 / 1024
    print(f"  wrote {mtx_path} ({sz:.1f} MB)")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("usage: csr_to_mtx.py <in.csr> <out.mtx>", file=sys.stderr)
        sys.exit(2)
    convert(Path(sys.argv[1]), Path(sys.argv[2]))
