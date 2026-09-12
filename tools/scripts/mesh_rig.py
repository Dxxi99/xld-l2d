#!/usr/bin/env python3
"""mesh_rig.py — anchor-based rotation rig for layered mesh.
yaw/pitch → per-layer offset → Laplacian-smoothed deform.
"""
import json
import numpy as np
from pathlib import Path
from skimage import io
from scipy import ndimage
from scipy.sparse import csr_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation

cache = Path("assets/real/swift_cache")
meta = open(cache/"meta.txt").read().split()
H, W = int(meta[1]), int(meta[3])
layer_id = np.fromfile(cache/"expected_layer_id.bin", dtype=np.uint8).reshape(H, W)

# --- Sample vertices ---
char_mask = layer_id < 7
ys, xs = np.where(char_mask)
stride = 12
ys_s = ys[::stride]; xs_s = xs[::stride]
n_v = len(ys_s)
v_layer = layer_id[ys_s, xs_s].astype(np.int32)
print(f"[grid] {n_v} verts")

# --- Triangulation ---
tri = Triangulation(xs_s, ys_s)
triangles = tri.triangles
cx = np.clip(xs_s[triangles].mean(axis=1).astype(int), 0, W-1)
cy = np.clip(ys_s[triangles].mean(axis=1).astype(int), 0, H-1)
keep = char_mask[cy, cx]
triangles = triangles[keep]
print(f"[tri] {len(triangles)} triangles")

# --- Laplacian matrix ---
e1 = np.stack([triangles[:,0], triangles[:,1]], 1)
e2 = np.stack([triangles[:,1], triangles[:,2]], 1)
e3 = np.stack([triangles[:,2], triangles[:,0]], 1)
edges = np.sort(np.concatenate([e1, e2, e3], 0), axis=1)
edges = np.unique(edges, axis=0)
A = csr_matrix((np.ones(len(edges), np.float32), (edges[:,0], edges[:,1])),
               shape=(n_v, n_v))
A = A + A.T
deg_inv = 1.0 / np.maximum(np.asarray(A.sum(axis=1)).ravel(), 1)
print(f"[adj] {len(edges)} edges")

# ============ ANCHORS & PIVOTS ============
# layer centroids
anchors = np.zeros((7, 2), np.float32)
for l in range(7):
    m = layer_id == l
    if m.sum() == 0:
        anchors[l] = [W/2, H/2]
        continue
    yy, xx = np.where(m)
    anchors[l] = [xx.mean(), yy.mean()]
print("[anchors]")
for l in range(7):
    print(f"  L{l}: ({anchors[l,0]:.0f}, {anchors[l,1]:.0f})")

# Pivots: rotation center per layer
# Layer roles (from semantic intuition):
#   L0-L3: head region (hair back / hair mid / face) → pivot = neck
#   L4-L5: body region → pivot = waist
#   L6:    legs/shoes → pivot = hip
neck_pivot = np.array([W/2, H*0.55], np.float32)
waist_pivot = np.array([W/2, H*0.70], np.float32)
hip_pivot = np.array([W/2, H*0.80], np.float32)

pivots = np.array([
    neck_pivot,   # L0
    neck_pivot,   # L1
    neck_pivot,   # L2
    neck_pivot,   # L3
    waist_pivot,  # L4
    waist_pivot,  # L5
    hip_pivot,    # L6
], dtype=np.float32)
print(f"[pivots] neck={neck_pivot}, waist={waist_pivot}, hip={hip_pivot}")

# Distance from each layer anchor to its pivot
layer_dist = np.linalg.norm(anchors - pivots, axis=1)
print(f"[dist] anchor-pivot distances:")
for l in range(7):
    print(f"  L{l}: {layer_dist[l]:.0f}px")

# ============ RIG ============
def compute_layer_offsets(yaw_deg, pitch_deg):
    """Return (7, 2) offsets for each layer given head rotation angles."""
    yaw = np.radians(yaw_deg)
    pitch = np.radians(pitch_deg)
    # horizontal offset: d * sin(yaw) — front layer moves more
    # depth-based influence: front layer (high z) moves more than back
    layer_z = np.array([0.038, 0.112, 0.178, 0.229, 0.274, 0.363, 0.463])
    z_norm = (layer_z - layer_z.min()) / (layer_z.max() - layer_z.min() + 1e-6)
    # front layers (z high) move more in +x for positive yaw
    influence = (z_norm - 0.5) * 2  # -1 (back) to +1 (front)

    offsets = np.zeros((7, 2), np.float32)
    for l in range(7):
        # rotational displacement around pivot
        # yaw: horizontal displacement
        off_x = influence[l] * 300 * np.sin(yaw)
        # pitch: vertical displacement (nod)
        off_y = influence[l] * 200 * (1 - np.cos(pitch))
        offsets[l] = [off_x, off_y]
    return offsets

def smooth_offsets(off_per_vertex, A, deg_inv, n_iter=60, alpha=0.5):
    off = off_per_vertex.copy()
    for _ in range(n_iter):
        nb_mean = (A @ off) * deg_inv[:, None]
        off = (1 - alpha) * off + alpha * nb_mean
    return off

# ============ RENDER GRID ============
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
cases = [
    (0,   0,   "idle"),
    (-20, 0,   "yaw -20°"),
    (20,  0,   "yaw +20°"),
    (0,  -15,  "pitch -15° (up)"),
    (0,   15,  "pitch +15° (down)"),
    (30,  20,  "yaw+pitch"),
]
for ax, (yaw, pitch, title) in zip(axes.ravel(), cases):
    layer_off = compute_layer_offsets(yaw, pitch)
    off_v = layer_off[v_layer].astype(np.float32)
    if yaw != 0 or pitch != 0:
        off_v = smooth_offsets(off_v, A, deg_inv, n_iter=60, alpha=0.5)
    xs_d = xs_s + off_v[:, 0]
    ys_d = ys_s + off_v[:, 1]
    ax.tripcolor(xs_d, ys_d, triangles,
                 facecolors=v_layer[triangles].mean(axis=1),
                 cmap='turbo', vmin=0, vmax=6, edgecolors='none', alpha=0.9)
    ax.set_xlim(-100, W+100); ax.set_ylim(H+100, -100)
    ax.set_aspect('equal')
    ax.set_title(title, fontsize=10)

plt.tight_layout()
out = Path("assets/real/mesh"); out.mkdir(parents=True, exist_ok=True)
plt.savefig(out / "deform_rig.png", dpi=80)
plt.close()
print(f"[save] {out/'deform_rig.png'}")

# ============ SAVE RIG CONFIG (for Swift) ============
rig_config = {
    'n_layers': 7,
    'anchors': anchors.tolist(),
    'pivots': pivots.tolist(),
    'layer_dist': layer_dist.tolist(),
    'layer_z': [0.038, 0.112, 0.178, 0.229, 0.274, 0.363, 0.463],
    'rig_x_scale': 300.0,
    'rig_y_scale': 200.0,
    'pitch_scale': 200.0,
    'smooth_iters': 60,
    'smooth_alpha': 0.5,
}
with open(out / "rig_config.json", "w") as f:
    json.dump(rig_config, f, indent=2)
print(f"[save] {out/'rig_config.json'}")

# Also dump vertices + weights for Swift
np.savez_compressed(
    out / "mesh_rigged.npz",
    xs=xs_s.astype(np.float32),
    ys=ys_s.astype(np.float32),
    v_layer=v_layer.astype(np.int32),
    triangles=triangles.astype(np.int32),
    anchors=anchors.astype(np.float32),
    pivots=pivots.astype(np.float32),
    layer_z=np.array(rig_config['layer_z'], np.float32),
    H=np.int32(H), W=np.int32(W),
)
print(f"[save] {out/'mesh_rigged.npz'}")
