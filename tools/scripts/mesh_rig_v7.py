#!/usr/bin/env python3
"""mesh_rig_v7.py — Lipschitz 강제 투영으로 flip=0 보장"""
import numpy as np
from pathlib import Path
from scipy.sparse import coo_matrix, diags
import sys

# ---- load v6 σ=80 (부드러운 시작점) ----
SRC = Path.home() / "XLD-L2D/assets/real/mesh/rig_v6_sig80_lam5.npz"
d = np.load(SRC)
xs = d['xs'].astype(np.float64)
ys = d['ys'].astype(np.float64)
tris = d['triangles'].astype(np.int64)
bx0 = d['basis_x'].astype(np.float64)
by0 = d['basis_y'].astype(np.float64)
R_X = float(d['R_X']); R_Y = float(d['R_Y'])
H, W = int(d['H']), int(d['W'])
n_v, n_t = len(xs), len(tris)
print(f"[load] n_v={n_v} n_t={n_t} R_X={R_X} R_Y={R_Y}")

# ---- unique undirected edges ----
e_dir = np.vstack([tris[:,[0,1]], tris[:,[1,2]], tris[:,[2,0]]])
key = np.sort(e_dir, axis=1)
eu = np.unique(key, axis=0)
n_e = len(eu)
print(f"[edges] unique = {n_e}")

el = np.linalg.norm(np.stack([xs[eu[:,0]]-xs[eu[:,1]], ys[eu[:,0]]-ys[eu[:,1]]],1), axis=1)
print(f"[edge len] mean={el.mean():.3f} min={el.min():.3f} p1={np.percentile(el,1):.3f} p5={np.percentile(el,5):.3f}")

# ---- pose budget ----
MAX_YAW   = float(sys.argv[1]) if len(sys.argv) > 1 else 30.0
MAX_PITCH = float(sys.argv[2]) if len(sys.argv) > 2 else 15.0
BUDGET    = float(sys.argv[3]) if len(sys.argv) > 3 else 0.45  # safety margin
sx_max = R_X * np.sin(np.deg2rad(MAX_YAW))
sy_max = R_Y * np.sin(np.deg2rad(MAX_PITCH))
print(f"[pose] yaw_max={MAX_YAW}° -> sx_max={sx_max:.2f}px")
print(f"[pose] pitch_max={MAX_PITCH}° -> sy_max={sy_max:.2f}px")
print(f"[budget] {BUDGET} per axis")

lim_x = el * BUDGET / sx_max
lim_y = el * BUDGET / sy_max

def project(basis, lim, n_iter=500, alpha=0.4, tol=1e-6):
    b = basis.copy()
    for it in range(n_iter):
        d_ = b[eu[:,0]] - b[eu[:,1]]
        over = np.abs(d_) - lim
        mask = over > 0
        n_viol = int(mask.sum())
        if n_viol == 0:
            print(f"  converged at iter {it} (0 violations)")
            return b, 0
        excess = over[mask] * 0.5
        sgn = np.sign(d_[mask])
        # Jacobi-style scatter (with np.add.at 안전)
        np.add.at(b, eu[mask,0], -alpha * sgn * excess)
        np.add.at(b, eu[mask,1], +alpha * sgn * excess)
        if it % 50 == 0:
            print(f"  iter {it}: violations={n_viol}  max_over={over.max():.4f}")
    print(f"  [warn] not converged, {n_viol} violations left")
    return b, n_viol

print("\n[project basis_x]")
bx, nv_x = project(bx0, lim_x)
print("\n[project basis_y]")
by, nv_y = project(by0, lim_y)

# ---- verify ----
def count_flips(x, y, t):
    p0 = np.stack([x[t[:,0]], y[t[:,0]]], 1)
    p1 = np.stack([x[t[:,1]], y[t[:,1]]], 1)
    p2 = np.stack([x[t[:,2]], y[t[:,2]]], 1)
    cr = (p1[:,0]-p0[:,0])*(p2[:,1]-p0[:,1]) - (p1[:,1]-p0[:,1])*(p2[:,0]-p0[:,0])
    return int((cr < 0).sum())

print("\n[flip grid]")
print(f"{'yaw\\p':>6}", "".join(f"{p:>8}" for p in [-15,0,15]))
total = 0
for yaw in [-30,-20,-10,0,10,20,30]:
    row = []
    for pitch in [-15,0,15]:
        sx = np.sin(np.deg2rad(yaw)) * R_X
        sy = np.sin(np.deg2rad(pitch)) * R_Y
        nf = count_flips(xs + bx*sx, ys + by*sy, tris)
        row.append(nf); total += nf
    print(f"{yaw:>6}", "".join(f"{v:>8}" for v in row))
print(f"[total flips] {total}")

print(f"\n[basis_x] min={bx.min():.4f} max={bx.max():.4f} range={bx.max()-bx.min():.4f}")
print(f"[basis_y] min={by.min():.4f} max={by.max():.4f} range={by.max()-by.min():.4f}")

out = Path.home() / "XLD-L2D/assets/real/mesh/rig_v7_flipfree.npz"
np.savez(out, xs=xs, ys=ys, triangles=tris,
         basis_x=bx.astype(np.float32), basis_y=by.astype(np.float32),
         H=H, W=W, R_X=R_X, R_Y=R_Y,
         MAX_YAW=MAX_YAW, MAX_PITCH=MAX_PITCH, BUDGET=BUDGET)
print(f"[save] {out}")
