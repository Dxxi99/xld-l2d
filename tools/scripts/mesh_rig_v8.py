#!/usr/bin/env python3
"""mesh_rig_v8.py — Gauss-Seidel Lipschitz projection, flip-free

Usage:
    python3 mesh_rig_v8.py [SRC_npz] [DEPTH_png] [DST_npz] [SIGMA] [LAMBDA] [R_X] [BUDGET]

Defaults:
    SRC_npz   = assets/real/mesh/rig_v3.npz
    DEPTH_png = assets/real/depth_anything_v2.png
    DST_npz   = assets/real/mesh/rig_v8_sig120_R100.npz
    SIGMA     = 120.0
    LAMBDA    = 20.0
    R_X       = 100.0
    BUDGET    = 0.50
"""
import sys
import numpy as np
from pathlib import Path
from PIL import Image
from scipy.ndimage import gaussian_filter
from scipy.sparse import coo_matrix, diags, identity
from scipy.sparse.linalg import spsolve

# ---- CLI ----
SRC_ARG   = sys.argv[1] if len(sys.argv) > 1 else None
DEPTH_ARG = sys.argv[2] if len(sys.argv) > 2 else None
DST_ARG   = sys.argv[3] if len(sys.argv) > 3 else None
SIGMA     = float(sys.argv[4]) if len(sys.argv) > 4 else 120.0
LAM       = float(sys.argv[5]) if len(sys.argv) > 5 else 20.0
R_X       = float(sys.argv[6]) if len(sys.argv) > 6 else 100.0
R_Y       = R_X * 0.6
BUDGET    = float(sys.argv[7]) if len(sys.argv) > 7 else 0.50

SRC   = Path(SRC_ARG).expanduser()   if SRC_ARG   else (Path.home() / "XLD-L2D/assets/real/mesh/rig_v3.npz")
DEPTH = Path(DEPTH_ARG).expanduser() if DEPTH_ARG else (Path.home() / "XLD-L2D/assets/real/depth_anything_v2.png")
DST   = Path(DST_ARG).expanduser()   if DST_ARG   else (Path.home() / "XLD-L2D/assets/real/mesh/rig_v8_sig120_R100.npz")

if not SRC.exists():   print(f"[error] SRC not found: {SRC}"); sys.exit(1)
if not DEPTH.exists(): print(f"[error] DEPTH not found: {DEPTH}"); sys.exit(1)
DST.parent.mkdir(parents=True, exist_ok=True)

print(f"[config] SRC={SRC}")
print(f"[config] DEPTH={DEPTH}")
print(f"[config] DST={DST}")
print(f"[config] SIGMA={SIGMA} LAMBDA={LAM} R_X={R_X} R_Y={R_Y} BUDGET={BUDGET}")

# ---- load mesh ----
d = np.load(SRC)
xs = d['xs'].astype(np.float64); ys = d['ys'].astype(np.float64)
tris = d['triangles'].astype(np.int64)
H, W = int(d['H']), int(d['W'])
n_v, n_t = len(xs), len(tris)
print(f"[load] n_v={n_v} n_t={n_t} H={H} W={W}")

# ---- unique edges + length ----
e_dir = np.vstack([tris[:,[0,1]], tris[:,[1,2]], tris[:,[2,0]]])
eu = np.unique(np.sort(e_dir, axis=1), axis=0)
el = np.linalg.norm(np.stack([xs[eu[:,0]]-xs[eu[:,1]], ys[eu[:,0]]-ys[eu[:,1]]],1), axis=1)
print(f"[edges] n={len(eu)} mean_len={el.mean():.3f}")

# ---- depth-based target ----
img = Image.open(DEPTH).convert("L")
if img.size != (W, H): img = img.resize((W, H), Image.BILINEAR)
depth = gaussian_filter(np.array(img, np.float32)/255.0, sigma=SIGMA)
xi = np.clip(xs.astype(np.int32), 0, W-1); yi = np.clip(ys.astype(np.int32), 0, H-1)
vz = depth[yi, xi].astype(np.float64)

# ---- mesh Laplacian for polish ----
E = np.vstack([e_dir, e_dir[:,::-1]])
A = coo_matrix((np.ones(len(E)), (E[:,0], E[:,1])), shape=(n_v,n_v)).tocsr()
A.data[:] = 1.0
deg = np.asarray(A.sum(1)).ravel(); deg[deg==0]=1
L = diags(deg) - A

# ---- Laplacian solve for smooth basis ----
vz_n = vz - vz.mean(); vz_n /= (np.abs(vz_n).max()+1e-8)
bx0 = spsolve((identity(n_v) + LAM*L).tocsc(), vz_n)
bx0 /= (np.abs(bx0).max()+1e-8)
by0 = -(ys - ys.mean()); by0 /= (np.abs(by0).max()+1e-8)
by0 = spsolve((identity(n_v) + LAM*L).tocsc(), by0)
by0 /= (np.abs(by0).max()+1e-8)

# ---- limits ----
MAX_YAW, MAX_PITCH = 30.0, 15.0
sx_max = R_X * np.sin(np.deg2rad(MAX_YAW))
sy_max = R_Y * np.sin(np.deg2rad(MAX_PITCH))
lim_x = el * BUDGET / sx_max
lim_y = el * BUDGET / sy_max
print(f"[limit_x] mean={lim_x.mean():.5f}")
print(f"[limit_y] mean={lim_y.mean():.5f}")

# ---- Gauss-Seidel projection ----
def gs_project(b, lim, n_epochs=60):
    rng = np.random.default_rng(42)
    n_e = len(eu)
    u = eu[:,0].copy(); v = eu[:,1].copy()
    for ep in range(n_epochs):
        perm = rng.permutation(n_e)
        bu = b[u[perm]]; bv = b[v[perm]]
        d_ = bu - bv
        over = np.abs(d_) - lim[perm]
        mask = over > 0
        if not mask.any():
            print(f"  epoch {ep}: converged (0 violations)")
            return b, 0
        excess = over[mask] * 0.5
        sgn = np.sign(d_[mask])
        push = np.zeros(n_v); cnt = np.zeros(n_v)
        np.add.at(push, u[perm][mask], -sgn*excess)
        np.add.at(push, v[perm][mask], +sgn*excess)
        np.add.at(cnt,  u[perm][mask], 1.0)
        np.add.at(cnt,  v[perm][mask], 1.0)
        cnt[cnt==0] = 1
        b += 0.5 * push / cnt
    d_ = b[u] - b[v]
    n_viol = int((np.abs(d_) > lim).sum())
    print(f"  [end] {n_viol} violations")
    return b, n_viol

print("\n[project basis_x]")
bx, nvx = gs_project(bx0.copy(), lim_x)
print("[project basis_y]")
by, nvy = gs_project(by0.copy(), lim_y)

# ---- verify flips ----
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

np.savez(DST, xs=xs, ys=ys, triangles=tris,
         basis_x=bx.astype(np.float32), basis_y=by.astype(np.float32),
         H=H, W=W, R_X=R_X, R_Y=R_Y, SIGMA=SIGMA, LAMBDA=LAM, BUDGET=BUDGET)
print(f"\n[save] {DST}")
