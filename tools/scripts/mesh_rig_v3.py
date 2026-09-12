#!/usr/bin/env python3
"""mesh_rig_v3.py — smooth sign + basis precompute for real-time rig

Usage:
    python3 mesh_rig_v3.py [LAYER_ID_BIN] [META_txt] [DST_npz] [STRIDE]

Defaults:
    LAYER_ID_BIN = assets/real/swift_cache/expected_layer_id.bin
    META_txt     = assets/real/swift_cache/meta.txt
    DST_npz      = assets/real/mesh/rig_v3.npz
    STRIDE       = 12
"""
import json
import sys
import time
import numpy as np
from pathlib import Path
from scipy.sparse import csr_matrix
from matplotlib.tri import Triangulation

# ---- CLI ----
LAYER_ARG = sys.argv[1] if len(sys.argv) > 1 else None
META_ARG  = sys.argv[2] if len(sys.argv) > 2 else None
DST_ARG   = sys.argv[3] if len(sys.argv) > 3 else None
STRIDE    = int(sys.argv[4]) if len(sys.argv) > 4 else 12

LAYER_PATH = Path(LAYER_ARG).expanduser() if LAYER_ARG else (
    Path.home() / "XLD-L2D/assets/real/swift_cache/expected_layer_id.bin")
META_PATH = Path(META_ARG).expanduser() if META_ARG else (
    Path.home() / "XLD-L2D/assets/real/swift_cache/meta.txt")
DST = Path(DST_ARG).expanduser() if DST_ARG else (
    Path.home() / "XLD-L2D/assets/real/mesh/rig_v3.npz")

if not LAYER_PATH.exists(): print(f"[error] layer not found: {LAYER_PATH}"); sys.exit(1)
if not META_PATH.exists():  print(f"[error] meta not found: {META_PATH}"); sys.exit(1)
DST.parent.mkdir(parents=True, exist_ok=True)
print(f"[cfg] LAYER={LAYER_PATH}")
print(f"[cfg] META={META_PATH}")
print(f"[cfg] DST={DST}  STRIDE={STRIDE}")

# ---- load meta + layer map ----
meta = open(META_PATH).read().split()
H, W = int(meta[1]), int(meta[3])
layer_id = np.fromfile(LAYER_PATH, dtype=np.uint8).reshape(H, W)
print(f"[meta] H={H} W={W}")

# ---- grid vertices ----
char_mask = layer_id < 7
ys, xs = np.where(char_mask)
ys_s = ys[::STRIDE]; xs_s = xs[::STRIDE]
n_v = len(ys_s)
v_layer = layer_id[ys_s, xs_s].astype(np.int32)
print(f"[grid] n_v={n_v}")

# ---- triangulation ----
tri = Triangulation(xs_s, ys_s)
triangles = tri.triangles
cx = np.clip(xs_s[triangles].mean(axis=1).astype(int), 0, W-1)
cy = np.clip(ys_s[triangles].mean(axis=1).astype(int), 0, H-1)
keep = char_mask[cy, cx]
triangles = triangles[keep]
print(f"[tri] {len(triangles)} triangles")

# ---- graph Laplacian ----
e1 = np.stack([triangles[:,0], triangles[:,1]], 1)
e2 = np.stack([triangles[:,1], triangles[:,2]], 1)
e3 = np.stack([triangles[:,2], triangles[:,0]], 1)
edges = np.sort(np.concatenate([e1, e2, e3], 0), axis=1)
edges = np.unique(edges, axis=0)
A = csr_matrix((np.ones(len(edges), np.float32), (edges[:,0], edges[:,1])),
               shape=(n_v, n_v))
A = A + A.T
deg_inv = 1.0 / np.maximum(np.asarray(A.sum(axis=1)).ravel(), 1)

# ---- smooth sign field ----
SIGN_PER_LAYER = np.array([-1.0, -1.0, +0.3, +1.0, +0.3, +0.2, 0.0], np.float32)
GROUP_WEIGHT   = np.array([ 1.0,  1.0,  1.0,  1.0, 0.25, 0.10, 0.0], np.float32)

sign_v = SIGN_PER_LAYER[v_layer].copy()
gw_v   = GROUP_WEIGHT[v_layer].copy()
for _ in range(30):
    sign_v = 0.7*sign_v + 0.3*(A @ sign_v)*deg_inv
    gw_v   = 0.7*gw_v   + 0.3*(A @ gw_v)*deg_inv
print(f"[sign] after smoothing: min={sign_v.min():.3f} max={sign_v.max():.3f}")

basis_x = (sign_v * gw_v).astype(np.float32)

pitch_sign = np.where(v_layer <= 3, 1.0, np.where(v_layer == 6, 0.0, -0.3)).astype(np.float32)
for _ in range(30):
    pitch_sign = 0.7*pitch_sign + 0.3*(A @ pitch_sign)*deg_inv
basis_y = (pitch_sign * gw_v).astype(np.float32)

print(f"[basis] yaw: min={basis_x.min():.3f} max={basis_x.max():.3f}")
print(f"[basis] pitch: min={basis_y.min():.3f} max={basis_y.max():.3f}")

# ---- save ----
np.savez_compressed(
    DST,
    xs=xs_s.astype(np.float32),
    ys=ys_s.astype(np.float32),
    v_layer=v_layer.astype(np.int32),
    triangles=triangles.astype(np.int32),
    basis_x=basis_x,
    basis_y=basis_y,
    sign_v=sign_v.astype(np.float32),
    gw_v=gw_v.astype(np.float32),
    H=np.int32(H), W=np.int32(W),
)
print(f"[save] {DST}")

# ---- benchmark ----
R_X = 400.0; R_Y = 250.0
def rig_offsets(yaw_deg, pitch_deg):
    yaw = np.radians(yaw_deg); pitch = np.radians(pitch_deg)
    return np.stack([basis_x * (np.sin(yaw)*R_X), basis_y * (np.sin(pitch)*R_Y)], axis=1)

N = 200
t0 = time.time()
for i in range(N):
    _ = rig_offsets(i*0.3 - 30, i*0.1 - 10)
dt = (time.time() - t0)/N
print(f"[bench] rig_offsets: {dt*1000:.2f}ms → {1/dt:.0f} fps")
