#!/usr/bin/env python3
"""mesh_rig_v5.py — depth-based smooth parallax rig (flip-free)"""
import numpy as np
from pathlib import Path
from PIL import Image
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import spsolve

# ---- load mesh ----
P = Path.home() / "XLD-L2D/assets/real/mesh/rig_v3.npz"
d = np.load(P)
xs = d['xs'].astype(np.float64)
ys = d['ys'].astype(np.float64)
tris = d['triangles'].astype(np.int64)
v_layer = d['v_layer'].astype(np.int32)
H, W = int(d['H']), int(d['W'])
n_v, n_t = len(xs), len(tris)
print(f"[load] n_v={n_v} n_t={n_t}  H={H} W={W}")

# ---- load depth map (원본 크기로) ----
DP = Path.home() / "XLD-L2D/assets/real/depth_anything_v2.png"
img = Image.open(DP)
print(f"[depth] {DP.name}  size={img.size}  mode={img.mode}")
if img.size != (W, H):
    img = img.resize((W, H), Image.BILINEAR)
depth = np.array(img.convert("L"), dtype=np.float32) / 255.0
print(f"[depth] range [{depth.min():.3f}, {depth.max():.3f}]  mean {depth.mean():.3f}")

# ---- sample depth at each vertex ----
xi = np.clip(xs.astype(np.int32), 0, W-1)
yi = np.clip(ys.astype(np.int32), 0, H-1)
vz = depth[yi, xi].astype(np.float64)
print(f"[vz raw] min={vz.min():.3f} max={vz.max():.3f} mean={vz.mean():.3f}")

# ---- smooth vz lightly (kernel 1-ring) to kill raster noise ----
e = np.vstack([tris[:,[0,1]], tris[:,[1,2]], tris[:,[2,0]]])
e = np.vstack([e, e[:,::-1]])
A = coo_matrix((np.ones(len(e)), (e[:,0], e[:,1])), shape=(n_v,n_v)).tocsr()
A.data[:] = 1.0
deg = np.asarray(A.sum(1)).ravel(); deg[deg==0]=1
for _ in range(20):
    vz = 0.5*vz + 0.5*(A @ vz)/deg
print(f"[vz smooth] min={vz.min():.3f} max={vz.max():.3f}")

# ---- normalize to zero-mean unit-range ----
vz = vz - vz.mean()
vz = vz / (np.abs(vz).max() + 1e-8)
print(f"[vz norm] min={vz.min():.3f} max={vz.max():.3f}")

# ---- yaw basis: horizontal parallax from depth ----
# front (high depth) moves opposite of back for head turn
basis_x = -vz  # scale factor R_X handles magnitude

# ---- pitch basis: vertical shift from height above neck ----
cy = ys.mean()
ys_norm = (ys - cy) / (ys.max() - ys.min())
basis_y = -ys_norm  # top moves down, chin moves up on +pitch

# optional: smooth basis_y lightly (already smooth but harmless)
for _ in range(10):
    basis_y = 0.5*basis_y + 0.5*(A @ basis_y)/deg
basis_y = basis_y / (np.abs(basis_y).max() + 1e-8)

print(f"[basis_x] min={basis_x.min():.3f} max={basis_x.max():.3f} mean={basis_x.mean():.3f}")
print(f"[basis_y] min={basis_y.min():.3f} max={basis_y.max():.3f} mean={basis_y.mean():.3f}")

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
        nf = count_flips(xs + basis_x*sx, ys + basis_y*sy, tris)
        row.append(nf); total += nf
    print(f"{yaw:>6}", "".join(f"{v:>8}" for v in row))
print(f"[total flips] {total}")

# ---- save ----
out = Path.home() / "XLD-L2D/assets/real/mesh/rig_v5.npz"
np.savez(out, xs=xs, ys=ys, triangles=tris, v_layer=v_layer,
         basis_x=basis_x.astype(np.float32),
         basis_y=basis_y.astype(np.float32),
         vz=vz.astype(np.float32),
         H=H, W=W)
print(f"[save] {out}")
