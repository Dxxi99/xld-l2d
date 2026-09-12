#!/usr/bin/env python3
"""mesh_rig_v12.py — bad-triangle 국소 Laplacian smoothing (flip=0)

Usage:
    python3 mesh_rig_v12.py [SRC_npz] [DST_npz] [ALPHA] [N_ITER]

Defaults:
    SRC_npz = assets/real/mesh/rig_v8_sig120_R100.npz
    DST_npz = assets/real/mesh/rig_v12_flipfree.npz
    ALPHA   = 0.15
    N_ITER  = 3000
"""
import sys
import numpy as np
from pathlib import Path

# ---- CLI ----
SRC_ARG = sys.argv[1] if len(sys.argv) > 1 else None
DST_ARG = sys.argv[2] if len(sys.argv) > 2 else None
ALPHA   = float(sys.argv[3]) if len(sys.argv) > 3 else 0.15
N_ITER  = int(sys.argv[4]) if len(sys.argv) > 4 else 3000

SRC = Path(SRC_ARG).expanduser() if SRC_ARG else (
    Path.home() / "XLD-L2D/assets/real/mesh/rig_v8_sig120_R100.npz")
DST = Path(DST_ARG).expanduser() if DST_ARG else (
    Path.home() / "XLD-L2D/assets/real/mesh/rig_v12_flipfree.npz")

if not SRC.exists():
    print(f"[error] SRC not found: {SRC}")
    sys.exit(1)
DST.parent.mkdir(parents=True, exist_ok=True)

print(f"[cfg] SRC={SRC}")
print(f"[cfg] DST={DST}")
print(f"[cfg] ALPHA={ALPHA} N_ITER={N_ITER}")

# ---- load ----
d = np.load(SRC)
xs = d['xs'].astype(np.float64)
ys = d['ys'].astype(np.float64)
tris = d['triangles'].astype(np.int64)
bx = d['basis_x'].astype(np.float64).copy()
by = d['basis_y'].astype(np.float64).copy()
R_X = float(d['R_X']); R_Y = float(d['R_Y'])
H, W = int(d['H']), int(d['W'])
n_v, n_t = len(xs), len(tris)
print(f"[load] n_v={n_v} n_t={n_t} R_X={R_X} R_Y={R_Y}")

GRID = [(y, p) for y in [-30,-20,-10,0,10,20,30] for p in [-15,0,15]]

def tri_area(x2, y2):
    p0x, p0y = x2[tris[:,0]], y2[tris[:,0]]
    p1x, p1y = x2[tris[:,1]], y2[tris[:,1]]
    p2x, p2y = x2[tris[:,2]], y2[tris[:,2]]
    return (p1x-p0x)*(p2y-p0y) - (p1y-p0y)*(p2x-p0x)

def total_flips():
    tot = 0
    for yaw, pitch in GRID:
        sx = np.sin(np.deg2rad(yaw))*R_X
        sy = np.sin(np.deg2rad(pitch))*R_Y
        A = tri_area(xs+bx*sx, ys+by*sy)
        tot += int((A<0).sum())
    return tot

print(f"[start] total flips = {total_flips()}")

for it in range(N_ITER):
    total_bad = 0
    for yaw, pitch in GRID:
        sx = np.sin(np.deg2rad(yaw))*R_X
        sy = np.sin(np.deg2rad(pitch))*R_Y
        A = tri_area(xs+bx*sx, ys+by*sy)
        bad_idx = np.where(A < 0)[0]
        total_bad += len(bad_idx)
        for t in bad_idx:
            a, b_, c = tris[t]
            mx = (bx[a]+bx[b_]+bx[c])/3
            my = (by[a]+by[b_]+by[c])/3
            for v in (a, b_, c):
                bx[v] = (1-ALPHA)*bx[v] + ALPHA*mx
                by[v] = (1-ALPHA)*by[v] + ALPHA*my
    if it < 5 or it % 50 == 0:
        print(f"  iter {it}: total_bad={total_bad}")
    if total_bad == 0:
        print(f"[converged] iter {it}")
        break

final = total_flips()
print(f"\n[end] total flips = {final}")
print(f"[basis_x] min={bx.min():.4f} max={bx.max():.4f} range={bx.max()-bx.min():.4f}")
print(f"[basis_y] min={by.min():.4f} max={by.max():.4f} range={by.max()-by.min():.4f}")

print("\n[flip grid]")
print(f"{'yaw\\p':>6}", "".join(f"{p:>8}" for p in [-15,0,15]))
for yaw in [-30,-20,-10,0,10,20,30]:
    row = []
    for pitch in [-15,0,15]:
        sx = np.sin(np.deg2rad(yaw))*R_X
        sy = np.sin(np.deg2rad(pitch))*R_Y
        A = tri_area(xs+bx*sx, ys+by*sy)
        row.append(int((A<0).sum()))
    print(f"{yaw:>6}", "".join(f"{v:>8}" for v in row))

np.savez(DST, xs=xs, ys=ys, triangles=tris,
         basis_x=bx.astype(np.float32), basis_y=by.astype(np.float32),
         H=H, W=W, R_X=R_X, R_Y=R_Y)
print(f"\n[save] {DST}")
