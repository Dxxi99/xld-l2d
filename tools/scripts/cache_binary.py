#!/usr/bin/env python3
"""cache_binary.py — dump cache.npz → raw .bin files Swift can read."""
import numpy as np
from pathlib import Path

src = Path("assets/real/swift_cache/cache.npz")
dst = Path("assets/real/swift_cache"); dst.mkdir(parents=True, exist_ok=True)

z = np.load(src)
H = int(z['H']); W = int(z['W']); n_seg = int(z['n_seg'])

# char_mask: uint8 (H*W)
cm = z['char_mask'].astype(np.uint8)
cm.tofile(dst / "char_mask.bin")

# seg_lbl: int32 (H*W)
sl = z['seg_lbl'].astype(np.int32)
sl.tofile(dst / "seg_lbl.bin")

# seg_z: float32 (n_seg+1)
sz = z['seg_z'].astype(np.float32)
sz.tofile(dst / "seg_z.bin")

# cons: int32 (N*3)
cs = z['cons'].astype(np.int32).ravel()
cs.tofile(dst / "cons.bin")

# nearest_seg: int32 (H*W) — the segment id at each pixel
ns = z['seg_lbl'][z['nearest_seg_yx'][...,0], z['nearest_seg_yx'][...,1]].astype(np.int32)
ns.tofile(dst / "nearest_seg.bin")

# metadata
with open(dst / "meta.txt", "w") as f:
    f.write(f"H {H}\nW {W}\nn_seg {n_seg}\nn_cons {len(z['cons'])}\n")

# expected z_n
exp = np.load(dst / "expected.npz")
exp['z_n'].astype(np.float32).tofile(dst / "expected_z_n.bin")
exp['topo_order'].astype(np.int32).tofile(dst / "expected_topo.bin")
exp['scc_lbl'].astype(np.int32).tofile(dst / "expected_scc_lbl.bin")

print(f"[bin] {dst}")
print(f"  char_mask.bin  {cm.nbytes} bytes")
print(f"  seg_lbl.bin    {sl.nbytes} bytes")
print(f"  seg_z.bin      {sz.nbytes} bytes")
print(f"  cons.bin       {cs.nbytes} bytes")
print(f"  nearest_seg.bin {ns.nbytes} bytes")
print(f"  expected_z_n.bin {exp['z_n'].nbytes} bytes")
