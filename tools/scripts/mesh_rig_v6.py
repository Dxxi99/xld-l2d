#!/usr/bin/env python3
"""mesh_rig_v6.py — image-space Gaussian smoothed depth basis (flip-free target)"""
import numpy as np
from pathlib import Path
from PIL import Image
from scipy.ndimage import gaussian_filter
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import spsolve
import sys

SIGMA = float(sys.argv[1]) if len(sys.argv) > 1 else 50.0
LAM   = float(sys.argv[2]) if len(sys.argv) > 2 else 5.0
R_X   = float(sys.argv[3]) if len(sys.argv) > 3 else 200.0
R_Y   = R_X * 0.6
print(f"[config] SIGMA={SIGMA} LAMBDA={LAM} R_X={R_X} R_Y={R_Y}")

# ---- load mesh ----
P = Path.home() / "XLD-L2D/assets/real/mesh/rig_v3.npz"
d = np.load(P)
xs = d['xs'].astype(np.float64)
ys = d['ys'].astype(np.float64)
tris = d['triangles'].astype(np.int64)
H, W = int(d['H']), int(d['W'])
n_v, n_t = len(xs), len(tris)

# ---- edge length stats ----
e = np.vstack([tris[:,[0,1]], tris[:,[1,2]], tris[:,[2,0]]])
el = np.linalg.norm(np.stack([xs[e[:,0]]-xs[e[:,1]], ys[e[:,0]]-ys[e[:,1]]],1), axis=1)
print(f"[edge] mean={el.mean():.2f}px  median={np.median(el):.2f}  p95={np.percentile(el,95):.2f}")

# ---- load & heavy smooth depth ----
DP = Path.home() / "XLD-L2D/assets/real/depth_anything_v2.png"
img = Image.open(DP).convert("L")
if img.size != (W, H):
    img = img.resize((W, H), Image.BILINEAR)
depth = np.array(img, dtype=np.float32) / 255.0
depth_s = gaussian_filter(depth, sigma=SIGMA)
print(f"[depth] raw [{depth.min():.3f},{depth.max():.3f}]  smoothed [{depth_s.min():.3f},{depth_s.max():.3f}]")

# ---- sample at verts ----
xi = np.clip(xs.astype(np.int32), 0, W-1)
yi = np.clip(ys.astype(np.int32), 0, H-1)
vz = depth_s[yi, xi].astype(np.float64)

# ---- mesh Laplacian for final polish ----
E = np.vstack([e, e[:,::-1]])
A = coo_matrix((np.ones(len(E)), (E[:,0], E[:,1])), shape=(n_v,n_v)).tocsr()
A.data[:] = 1.0
deg = np.asarray(A.sum(1)).ravel(); deg[deg==0]=1
L = diags(deg) - A

# normalize target to zero-mean
vz = vz - vz.mean()
scale = np.abs(vz).max() + 1e-8
vz = vz / scale
print(f"[vz target] min={vz.min():.3f} max={vz.max():.3f}")

# ---- Laplacian solve: (I + λL) b = target ----
from scipy.sparse import identity
M = (identity(n_v) + LAM * L).tocsc()
bx = spsolve(M, vz)
by_raw = -(ys - ys.mean()) / (ys.max() - ys.min())   # top-down for pitch
by = spsolve(M, by_raw)

# renormalize basis_x to [-1,1]  (scale 안 곱하는 대신 basis 자체를 [-1,1])
bx = bx / (np.abs(bx).max() + 1e-8)
by = by / (np.abs(by).max() + 1e-8)
print(f"[basis_x] min={bx.min():.3f} max={bx.max():.3f} mean={bx.mean():.3f}")
print(f"[basis_y] min={by.min():.3f} max={by.max():.3f}")

# ---- flip grid ----
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

out = Path.home() / "XLD-L2D/assets/real/mesh" / f"rig_v6_sig{int(SIGMA)}_lam{int(LAM)}.npz"
np.savez(out, xs=xs, ys=ys, triangles=tris,
         basis_x=bx.astype(np.float32), basis_y=by.astype(np.float32),
         H=H, W=W, R_X=R_X, R_Y=R_Y, SIGMA=SIGMA, LAMBDA=LAM)
print(f"[save] {out}")
