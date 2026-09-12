#!/usr/bin/env python3
import numpy as np
from pathlib import Path

SRC = Path.home() / "XLD-L2D/assets/real/mesh/rig_v12_flipfree.npz"
DST = Path.home() / "XLD-L2D/assets/real/swift_cache/rig_v12"
DST.mkdir(parents=True, exist_ok=True)

d = np.load(SRC)
n_v = len(d['xs'])
print(f"[src] {SRC}")
print(f"  n_v={n_v}  R_X={float(d['R_X'])}  R_Y={float(d['R_Y'])}  H={int(d['H'])}  W={int(d['W'])}")

d['basis_x'].astype(np.float32).tofile(DST / "basis_x.bin")
d['basis_y'].astype(np.float32).tofile(DST / "basis_y.bin")
with open(DST / "meta.txt", "w") as f:
    f.write(f"n_v {n_v}\n")
    f.write(f"R_X {float(d['R_X'])}\n")
    f.write(f"R_Y {float(d['R_Y'])}\n")
    f.write(f"H {int(d['H'])}\n")
    f.write(f"W {int(d['W'])}\n")

yaw_deg, pitch_deg = 25.0, 15.0
sx = np.sin(np.deg2rad(yaw_deg)) * float(d['R_X'])
sy = np.sin(np.deg2rad(pitch_deg)) * float(d['R_Y'])
off_x = (d['basis_x'].astype(np.float64) * sx).astype(np.float32)
off_y = (d['basis_y'].astype(np.float64) * sy).astype(np.float32)
off_x.tofile(DST / "expected_off_x_yaw25.bin")
off_y.tofile(DST / "expected_off_y_pitch15.bin")

print(f"[save] {DST}")
for f in sorted(DST.iterdir()):
    print(f"  {f.name}  {f.stat().st_size} bytes")
