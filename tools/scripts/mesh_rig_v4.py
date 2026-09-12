#!/usr/bin/env python3
"""mesh_rig_v4.py — Laplacian solve로 찢어짐 제거"""
import numpy as np
from pathlib import Path
from scipy.sparse import coo_matrix, identity, diags
from scipy.sparse.linalg import spsolve
import sys

# ---- config ----
LAMBDA = float(sys.argv[1]) if len(sys.argv) > 1 else 100.0
print(f"[config] LAMBDA = {LAMBDA}")

# ---- load ----
P = Path.home() / "XLD-L2D/assets/real/mesh/rig_v3.npz"
d = np.load(P)
xs = d['xs'].astype(np.float64)
ys = d['ys'].astype(np.float64)
tris = d['triangles'].astype(np.int64)
v_layer = d['v_layer'].astype(np.int32)
H, W = int(d['H']), int(d['W'])
n_v, n_t = len(xs), len(tris)
print(f"[load] n_v={n_v} n_t={n_t}")

# ---- graph Laplacian (uniform) ----
e = np.vstack([tris[:,[0,1]], tris[:,[1,2]], tris[:,[2,0]]])
e = np.vstack([e, e[:,::-1]])
A = coo_matrix((np.ones(len(e)), (e[:,0], e[:,1])), shape=(n_v,n_v)).tocsr()
A.data[:] = 1.0
deg = np.asarray(A.sum(1)).ravel()
L = diags(deg) - A          # graph Laplacian

# ---- target & confidence ----
sign_vec = np.array([-1, -1, +0.3, +1, +0.3, +0.2, 0], dtype=np.float64)
gw_vec   = np.array([ 1,  1,    1,  1, 0.25, 0.10, 0], dtype=np.float64)
conf_vec = np.array([1.0, 1.0, 0.9, 1.0, 0.5, 0.3, 0.05], dtype=np.float64)

target_x = (sign_vec * gw_vec)[v_layer]
target_y = (sign_vec * gw_vec)[v_layer]
w = conf_vec[v_layer]

# ---- solve (w + λL) b = w·target ----
I = identity(n_v, format='csr')
M = (diags(w) + LAMBDA * L).tocsc()
print("[solve] factorizing...")
bx = spsolve(M, w * target_x)
by = spsolve(M, w * target_y)
print(f"[basis_x] min={bx.min():.4f} max={bx.max():.4f}")
print(f"[basis_y] min={by.min():.4f} max={by.max():.4f}")

# ---- flip grid ----
def count_flips(x, y, t):
    p0 = np.stack([x[t[:,0]], y[t[:,0]]], 1)
    p1 = np.stack([x[t[:,1]], y[t[:,1]]], 1)
    p2 = np.stack([x[t[:,2]], y[t[:,2]]], 1)
    cr = (p1[:,0]-p0[:,0])*(p2[:,1]-p0[:,1]) - (p1[:,1]-p0[:,1])*(p2[:,0]-p0[:,0])
    return int((cr < 0).sum())

R_X, R_Y = 400.0, 250.0
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

# ---- save ----
out = Path.home() / "XLD-L2D/assets/real/mesh" / f"rig_v4_lambda{int(LAMBDA)}.npz"
np.savez(out, xs=xs, ys=ys, triangles=tris, v_layer=v_layer,
         basis_x=bx.astype(np.float32), basis_y=by.astype(np.float32),
         sign_v=sign_vec[v_layer].astype(np.float32),
         gw_v=gw_vec[v_layer].astype(np.float32),
         H=H, W=W, LAMBDA=LAMBDA)
print(f"[save] {out}")
